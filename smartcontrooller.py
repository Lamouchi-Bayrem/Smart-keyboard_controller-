#!/usr/bin/env python3
"""
multigame_smooth_keyboard.py
A smoother Smart Control App with non-blocking brightness control and an on-screen gesture keyboard.
Features:
 - Adaptive frame pacing and reduced MP input resolution for speed.
 - Background BrightnessWorker that coalesces and throttles brightness changes.
 - Two-hand mode: draws an on-camera keyboard when two hands detected; hover a key for >1s to type it.
 - Improved cursor smoothing and configurable parameters saved to JSON.
 - Minimal overlays and performance indicators.

Requirements: Python 3.8+, opencv-python, mediapipe, numpy, pyautogui, PyQt5
Optional: screen_brightness_control, pycaw (Windows audio)

"""

import sys
import os
import json
import time
import math
import platform
import subprocess
import logging
import threading
from queue import Queue
from collections import deque, defaultdict

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QCheckBox, QDoubleSpinBox,
    QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox, QFormLayout, QMessageBox,
    QSpinBox, QStyleFactory
)

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("smooth-app")

# -------------------- Settings --------------------
SETTINGS_FILE = "multigame_settings.json"
DEFAULTS = {
    "camera_index": 0,
    "camera_w": 640,
    "camera_h": 480,
    "camera_fps": 30,
    "mp_input_w": 320,
    "mp_input_h": 240,
    "target_fps": 30,
    "process_every_n": 1,
    "mirror": True,
    "draw_overlay": True,
    "minimal_overlay": True,
    "mouse_ema_alpha": 0.35,
    "mouse_vel_gate": 0.015,
    "control_ema_alpha": 0.25,
    "click_finger_distance": 0.04,
    "click_debounce_ms": 220,
    "drag_hold_s": 0.5,
    "brightness_threshold": 0.02,
    "brightness_throttle_hz": 8,
    "brightness_ignore_delta": 2,
    "brightness_change_speed": 1.0,
    "brightness_dead_zone": 0.03,
    "volume_throttle_hz": 8,
    "volume_ignore_delta": 2,
    "volume_change_speed": 1.0,
    "volume_dead_zone": 0.03,
    "keyboard_offset_x": 0,
    "keyboard_offset_y": 0,
    "failsafe": False,
    "show_help": True,
    "calib_A": None,
    "adaptive_pacing": True,
    "mouse_sensitivity": 1.0,
    "click_min_distance": 10,
    "click_hold_time_ms": 100,
    "click_smoothing_alpha": 0.4,
    "hand_switch_distance": 0.05,
    "pinch_threshold": 0.04,
    "control_mode_lock_ms": 500,
}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
            for k, v in DEFAULTS.items():
                if k not in data:
                    data[k] = v
            return data
        except Exception as e:
            logger.warning(f"Failed to load settings, using defaults: {e}")
            return DEFAULTS.copy()
    return DEFAULTS.copy()


def save_settings(s):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# -------------------- System helpers --------------------
class SystemVolume:
    def __init__(self):
        self.method = None
        self.cur = 0.5
        osname = platform.system()
        if osname == 'Windows':
            try:
                from ctypes import POINTER, cast
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                dev = AudioUtilities.GetSpeakers()
                iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(iface, POINTER(IAudioEndpointVolume))
                self.min_vol, self.max_vol, _ = self.volume.GetVolumeRange()
                self.method = 'pycaw'
                try:
                    db = self.volume.GetMasterVolumeLevel()
                    self.cur = (db - self.min_vol) / (self.max_vol - self.min_vol)
                except Exception:
                    pass
            except Exception:
                self.method = None
        elif osname == 'Linux':
            try:
                r = subprocess.run(['pactl', 'info'], capture_output=True, text=True, timeout=1)
                if r.returncode == 0:
                    self.method = 'pulse'
            except Exception:
                pass
        elif osname == 'Darwin':
            try:
                r = subprocess.run(['osascript', '-e', 'get volume settings'], capture_output=True, text=True, timeout=1)
                if r.returncode == 0:
                    self.method = 'osascript'
            except Exception:
                pass

    def set_norm(self, v):
        v = max(0.0, min(1.0, float(v)))
        try:
            if self.method == 'pycaw':
                db = self.min_vol + (self.max_vol - self.min_vol) * v
                self.volume.SetMasterVolumeLevel(db, None)
                self.cur = v
                return True
            elif self.method == 'pulse':
                subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f"{int(v*100)}%"], timeout=1)
                self.cur = v
                return True
            elif self.method == 'osascript':
                subprocess.run(['osascript', '-e', f'set volume output volume {int(v*100)}'], timeout=1)
                self.cur = v
                return True
        except Exception:
            pass
        try:
            steps = int(round((v - self.cur) * 10))
            if steps > 0:
                for _ in range(steps):
                    pyautogui.press('volumeup'); time.sleep(0.02)
            elif steps < 0:
                for _ in range(-steps):
                    pyautogui.press('volumedown'); time.sleep(0.02)
            self.cur = v
            return True
        except Exception:
            return False

sysvol = SystemVolume()

# -------------------- Non-blocking brightness worker --------------------
class BrightnessWorker(threading.Thread):
    """Coalesces brightness requests and applies them off the video thread.
    Applies the latest requested brightness at most 10x/sec and ignores tiny diffs.
    """
    def __init__(self, throttle_hz=10, ignore_delta=2, change_speed=1.0):
        super().__init__(daemon=True)
        self.throttle = max(1, throttle_hz)
        self.ignore_delta = ignore_delta
        self.change_speed = change_speed
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._applied = None
        self.start()

    def run(self):
        interval = 1.0 / self.throttle
        while not self._stop.is_set():
            time.sleep(interval)
            with self._lock:
                val = self._latest
                self._latest = None
            if val is None:
                continue
            if self._applied is not None:
                delta = val - self._applied
                adjusted_delta = delta * self.change_speed * (interval / 0.1)
                val = self._applied + adjusted_delta
                if abs(val - self._applied) < self.ignore_delta:
                    continue
            ok = self._apply_brightness(val)
            if ok:
                self._applied = val

    def set(self, percent):
        with self._lock:
            self._latest = int(max(0, min(100, percent)))

    def stop(self):
        self._stop.set()

    def _apply_brightness(self, percent):
        try:
            import screen_brightness_control as sbc
            sbc.set_brightness(percent)
            return True
        except Exception:
            pass
        osname = platform.system()
        try:
            if osname == 'Windows':
                cmd = f'powershell.exe "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{percent})"'
                subprocess.run(cmd, shell=True, timeout=2)
                return True
            elif osname == 'Linux':
                brightness_val = str(percent / 100.0)
                for d in ['eDP-1','LVDS-1','HDMI-1','VGA-1','DP-1']:
                    try:
                        r = subprocess.run(['xrandr','--output',d,'--brightness',brightness_val], capture_output=True, text=True, timeout=1)
                        if r.returncode == 0:
                            return True
                    except Exception:
                        continue
            elif osname == 'Darwin':
                subprocess.run(['brightness', str(percent/100.0)], timeout=2)
                return True
        except Exception:
            pass
        return False

# -------------------- Non-blocking volume worker --------------------
class VolumeWorker(threading.Thread):
    """Coalesces volume requests and applies them off the video thread.
    Applies the latest requested volume at most 10x/sec and ignores tiny diffs.
    """
    def __init__(self, throttle_hz=8, ignore_delta=2, change_speed=1.0):
        super().__init__(daemon=True)
        self.throttle = max(1, throttle_hz)
        self.ignore_delta = ignore_delta
        self.change_speed = change_speed
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._applied = None
        self.start()

    def run(self):
        interval = 1.0 / self.throttle
        while not self._stop.is_set():
            time.sleep(interval)
            with self._lock:
                val = self._latest
                self._latest = None
            if val is None:
                continue
            if self._applied is not None:
                delta = val - self._applied
                adjusted_delta = delta * self.change_speed * (interval / 0.1)
                val = self._applied + adjusted_delta
                if abs(val - self._applied) < self.ignore_delta:
                    continue
            ok = self._apply_volume(val)
            if ok:
                self._applied = val

    def set(self, percent):
        with self._lock:
            self._latest = int(max(0, min(100, percent)))

    def stop(self):
        self._stop.set()

    def _apply_volume(self, percent):
        return sysvol.set_norm(percent / 100.0)

bri_worker = None
vol_worker = None

# -------------------- MediaPipe wrapper --------------------
class MPHands:
    def __init__(self, max_num_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0,
            static_image_mode=False
        )
        self.drawing = mp.solutions.drawing_utils

    def process(self, rgb_small):
        return self.hands.process(rgb_small)

    def close(self):
        self.hands.close()

# -------------------- Filters --------------------
class CursorFilter:
    def __init__(self, alpha=0.35, vel_gate=0.015):
        self.alpha = float(alpha)
        self.gate = float(vel_gate)
        self.prev = None
        self.prev_time = None

    def step(self, x, y):
        now = time.time()
        if self.prev is None:
            self.prev = (x, y); self.prev_time = now; return x, y
        dt = max(1e-3, now - self.prev_time)
        vx = abs(x - self.prev[0]) / dt
        vy = abs(y - self.prev[1]) / dt
        speed = max(vx, vy)
        alpha = self.alpha * 0.5 if speed < self.gate else self.alpha
        sx = alpha * x + (1-alpha) * self.prev[0]
        sy = alpha * y + (1-alpha) * self.prev[1]
        self.prev = (sx, sy); self.prev_time = now
        return sx, sy

# -------------------- On-Screen Keyboard --------------------
class OnScreenKeyboard:
    def __init__(self, label_w, label_h, offset_x=0, offset_y=0):
        self.w = label_w; self.h = label_h
        self.offset_x = offset_x
        self.offset_y = offset_y
        rows = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM")]
        rows[2] = rows[2] + ['SPACE','BKSP','ENTER']
        self.rows = rows
        self.key_boxes = []
        self._compute_layout()
        self.hover_timers = {}
        self.hover_threshold = 1.0

    def _compute_layout(self):
        pad = 8
        key_h = int(self.h * 0.10)
        y0 = self.h - (len(self.rows) * (key_h + pad)) - 10 + self.offset_y
        self.key_boxes.clear()
        for r_i, row in enumerate(self.rows):
            rw = len(row)
            total_pad = pad * (rw + 1)
            key_w = int((self.w - total_pad) / rw)
            x = pad + self.offset_x
            ky = y0 + r_i * (key_h + pad)
            for k in row:
                self.key_boxes.append((x, ky, x+key_w, ky+key_h, k))
                x += key_w + pad

    def draw(self, frame):
        for (x0,y0,x1,y1,k) in self.key_boxes:
            cv2.rectangle(frame, (x0,y0), (x1,y1), (40,40,40), -1)
            cv2.rectangle(frame, (x0,y0), (x1,y1), (200,200,200), 1)
            txt = k if len(k) <= 4 else k[0:4]
            cv2.putText(frame, txt, (x0+6, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

    def hit_test(self, px, py):
        for (x0,y0,x1,y1,k) in self.key_boxes:
            if x0 <= px <= x1 and y0 <= py <= y1:
                return k, (x0,y0,x1,y1)
        return None, None

    def update_hover(self, key, now):
        if key is None:
            self.hover_timers.clear(); return False
        if key not in self.hover_timers:
            self.hover_timers[key] = now
            return False
        start = self.hover_timers.get(key)
        if start and (now - start) >= self.hover_threshold:
            self.hover_timers.pop(key, None)
            return True
        return False

# -------------------- Video Thread --------------------
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(dict)

    def __init__(self, settings):
        super().__init__()
        self.s = settings
        self.run_flag = True
        pyautogui.FAILSAFE = bool(self.s.get('failsafe', False))

        self.mp = MPHands(max_num_hands=2)
        self.cursor_filter = CursorFilter(alpha=self.s.get('mouse_ema_alpha',0.35), vel_gate=self.s.get('mouse_vel_gate',0.015))

        self.target_dt = 1.0 / max(1, int(self.s.get('target_fps',30)))
        self.process_every_n = max(1, int(self.s.get('process_every_n',1)))
        self._proc_counter = 0

        self.last_click_t = 0
        self.last_click_pos = None
        self.drag_start_t = None
        self.dragging = False

        self.last_brightness = None
        self.last_volume = sysvol.cur  # Initialize from system
        self.last_bri_update = 0
        self.last_vol_update = 0

        self.screen_w, self.screen_h = pyautogui.size()
        self.calib_A = None
        if self.s.get('calib_A'):
            try:
                self.calib_A = np.array(self.s['calib_A'], dtype=np.float32)
            except Exception:
                self.calib_A = None

        self.keyboard = None
        self.last_key_pressed_time = 0
        self.hover_feedback = None

        self.click_indicator_start = None
        self.click_indicator_duration = 0.5
        self.brightness_indicator_start = None
        self.volume_indicator_start = None
        self.indicator_duration = 1.0

        self.current_idx_pos = None
        self.adaptive_pacing = self.s.get('adaptive_pacing', True)
        self.mouse_sensitivity = self.s.get('mouse_sensitivity', 1.0)
        self.click_min_distance = self.s.get('click_min_distance', 10)
        self.click_hold_time_ms = self.s.get('click_hold_time_ms', 100)
        self.click_smoothing_alpha = self.s.get('click_smoothing_alpha', 0.4)
        self.click_finger_distance = self.s.get('click_finger_distance', 0.04)
        self.hand_switch_distance = self.s.get('hand_switch_distance', 0.05)

        self.fingers_close_start = None
        self.fingers_close_smoothed = 0.0
        self.active_hand = 0
        self.index_fingers_close_start = None
        self.index_fingers_close_smoothed = 0.0
        self.switch_indicator_start = None

        # New for control mode
        self.control_mode = None  # "brightness" or "volume"
        self.control_lock_until = 0
        self.control_mode_lock_ms = self.s.get('control_mode_lock_ms', 500)
        self.brightness_dead_zone = self.s.get('brightness_dead_zone', 0.03)
        self.volume_dead_zone = self.s.get('volume_dead_zone', 0.03)
        self.control_rate_limit = 0.08  # ~12.5 Hz

    def map_to_screen(self, lm_x, lm_y):
        x = 0.5 + self.mouse_sensitivity * (lm_x - 0.5)
        y = 0.5 + self.mouse_sensitivity * (lm_y - 0.5)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        if self.calib_A is not None:
            v = np.array([x, y, 1.0])
            proj = self.calib_A @ v
            sx = proj[0] / proj[2]
            sy = proj[1] / proj[2]
            return int(np.clip(sx, 0, self.screen_w-1)), int(np.clip(sy, 0, self.screen_h-1))
        sx = int(x * self.screen_w); sy = int(y * self.screen_h)
        return sx, sy

    def adaptive_backoff(self, loop_time):
        if loop_time > self.target_dt * 1.3 and self.process_every_n < 4:
            self.process_every_n += 1
        elif loop_time < self.target_dt * 0.7 and self.process_every_n > 1:
            self.process_every_n -= 1

    def run(self):
        cap = cv2.VideoCapture(int(self.s.get('camera_index',0)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.s.get('camera_w',640)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.s.get('camera_h',480)))
        cap.set(cv2.CAP_PROP_FPS, int(self.s.get('camera_fps',30)))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error("Camera open failed")
            return

        mp_w = int(self.s.get('mp_input_w',320)); mp_h = int(self.s.get('mp_input_h',240))

        fps_cnt, fps_t0 = 0, time.time(); current_fps = 0

        while self.run_flag:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                continue
            if self.s.get('mirror', True):
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if self.keyboard is None:
                self.keyboard = OnScreenKeyboard(w, h, self.s.get('keyboard_offset_x', 0), self.s.get('keyboard_offset_y', 0))

            self._proc_counter = (self._proc_counter + 1) % max(1, self.process_every_n)
            results = None
            if self._proc_counter == 0:
                small = cv2.resize(frame, (mp_w, mp_h))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                try:
                    results = self.mp.process(rgb)
                except Exception:
                    results = None

            status = {"mode": "Idle", "fps": current_fps, "volume": None, "brightness": None}

            self.current_idx_pos = None

            if results and results.multi_hand_landmarks:
                hands = list(zip(results.multi_hand_landmarks, results.multi_handedness))
                if self.s.get('draw_overlay', True):
                    for i, (hlm, _) in enumerate(hands):
                        tip_i = hlm.landmark[8]
                        cx = int(tip_i.x * w); cy = int(tip_i.y * h)
                        color = (0, 255, 0) if i == self.active_hand else (255, 0, 0)
                        cv2.circle(frame, (cx, cy), 6, color, -1)

                if len(hands) >= 2:
                    status['mode'] = 'Keyboard'
                    hlm0, info0 = hands[0]
                    hlm1, info1 = hands[1]
                    idx0 = hlm0.landmark[8]
                    idx1 = hlm1.landmark[8]
                    index_dist = math.hypot(idx0.x - idx1.x, idx0.y - idx1.y)
                    index_close_raw = 1.0 if index_dist < self.hand_switch_distance else 0.0
                    self.index_fingers_close_smoothed = self.click_smoothing_alpha * index_close_raw + (1 - self.click_smoothing_alpha) * self.index_fingers_close_smoothed
                    index_fingers_close = self.index_fingers_close_smoothed > 0.5
                    now = time.time()
                    if index_fingers_close:
                        if self.index_fingers_close_start is None:
                            self.index_fingers_close_start = now
                        elif (now - self.index_fingers_close_start) * 1000 >= self.click_hold_time_ms:
                            self.active_hand = 1 - self.active_hand
                            self.index_fingers_close_start = None
                            self.switch_indicator_start = now
                            status['hand_switched'] = self.active_hand
                    else:
                        self.index_fingers_close_start = None

                    hlm_active = hands[self.active_hand][0]
                    idx_active = hlm_active.landmark[8]
                    px = int(idx_active.x * w); py = int(idx_active.y * h)
                    if self.s.get('draw_overlay', True):
                        self.keyboard.draw(frame)
                        cv2.circle(frame, (px, py), 10, (0, 255, 0), 2)
                    key, box = self.keyboard.hit_test(px, py)
                    selected = False
                    if key:
                        x0, y0, x1, y1 = box
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 160, 0), -1)
                        cv2.putText(frame, key if len(key) <= 4 else key[0:4], (x0+6, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                        if self.keyboard.update_hover(key, now):
                            selected = True
                    else:
                        self.keyboard.update_hover(None, now)

                    if selected:
                        if key == 'SPACE':
                            pyautogui.press('space')
                        elif key == 'BKSP':
                            pyautogui.press('backspace')
                        elif key == 'ENTER':
                            pyautogui.press('enter')
                        else:
                            pyautogui.typewrite(key.lower())
                        status['typed'] = key
                        time.sleep(0.15)
                else:
                    hlm, hinfo = hands[0]
                    label = hinfo.classification[0].label
                    lm = hlm.landmark
                    if label == 'Right':
                        status['mode'] = 'Mouse'
                        idx = lm[8]; mid = lm[12]; th = lm[4]
                        self.current_idx_pos = (idx.x, idx.y)
                        sx, sy = self.map_to_screen(idx.x, idx.y)
                        sx, sy = self.cursor_filter.step(sx, sy)
                        sx = int(np.clip(sx, 10, self.screen_w-10)); sy = int(np.clip(sy, 10, self.screen_h-10))
                        try:
                            pyautogui.moveTo(sx, sy)
                        except Exception:
                            pass
                        finger_dist = math.hypot(idx.x - mid.x, idx.y - mid.y)
                        fingers_close_raw = 1.0 if finger_dist < self.click_finger_distance else 0.0
                        self.fingers_close_smoothed = self.click_smoothing_alpha * fingers_close_raw + (1 - self.click_smoothing_alpha) * self.fingers_close_smoothed
                        fingers_close = self.fingers_close_smoothed > 0.5
                        now = time.time()
                        if fingers_close:
                            if self.fingers_close_start is None:
                                self.fingers_close_start = now
                        else:
                            self.fingers_close_start = None
                        can_click = True
                        if self.last_click_pos is not None:
                            dist = math.hypot(sx - self.last_click_pos[0], sy - self.last_click_pos[1])
                            if dist < self.click_min_distance:
                                can_click = False
                        if fingers_close and not self.dragging and can_click and self.fingers_close_start and (now - self.fingers_close_start) * 1000 >= self.click_hold_time_ms and (now - self.last_click_t) * 1000 > int(self.s.get('click_debounce_ms', 220)):
                            try:
                                pyautogui.click(button='left')
                                self.last_click_t = now
                                self.last_click_pos = (sx, sy)
                                self.click_indicator_start = now
                            except Exception:
                                pass
                        pinch = math.hypot(th.x - idx.x, th.y - idx.y)
                        if pinch < self.s.get('pinch_threshold', 0.04):
                            if self.drag_start_t is None:
                                self.drag_start_t = now
                            elif (now - self.drag_start_t) >= self.s.get('drag_hold_s', 0.5) and not self.dragging:
                                try:
                                    pyautogui.mouseDown()
                                    self.dragging = True
                                except Exception:
                                    pass
                        else:
                            if self.dragging:
                                try:
                                    pyautogui.mouseUp()
                                    self.dragging = False
                                except Exception:
                                    pass
                            self.drag_start_t = None
                    else:
                        status['mode'] = 'Control'
                        th = lm[4]; idx = lm[8]
                        dist = math.hypot(th.x-idx.x, th.y-idx.y)
                        dx = idx.x - th.x; dy = idx.y - th.y
                        angle = math.degrees(math.atan2(abs(dy), abs(dx))) if (dx or dy) else 0
                        now = time.time()
                        if now < self.control_lock_until:
                            vertical = self.control_mode == "brightness"
                        else:
                            vertical = angle > 45
                            self.control_mode = "brightness" if vertical else "volume"
                            self.control_lock_until = now + self.control_mode_lock_ms / 1000.0
                        val = float(np.clip((dist - 0.02)/0.15, 0, 1))
                        alpha = float(self.s.get('control_ema_alpha', 0.25))
                        if vertical:
                            new_brightness = val if self.last_brightness is None else alpha*val + (1-alpha)*self.last_brightness
                            if abs(new_brightness - (self.last_brightness or 0)) > self.brightness_dead_zone and now - self.last_bri_update > self.control_rate_limit:
                                self.last_brightness = new_brightness
                                bri_worker.set(int(self.last_brightness*100))
                                self.last_bri_update = now
                                self.brightness_indicator_start = now
                            status['brightness'] = int((self.last_brightness or 0)*100)
                            if self.s.get('draw_overlay', True):
                                cv2.putText(frame, "Brightness Control", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        else:
                            new_volume = val if self.last_volume is None else alpha*val + (1-alpha)*self.last_volume
                            if abs(new_volume - (self.last_volume or 0)) > self.volume_dead_zone and now - self.last_vol_update > self.control_rate_limit:
                                self.last_volume = new_volume
                                vol_worker.set(int(self.last_volume * 100))
                                self.last_vol_update = now
                                self.volume_indicator_start = now
                            status['volume'] = int((self.last_volume or 0)*100)
                            if self.s.get('draw_overlay', True):
                                cv2.putText(frame, "Volume Control", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            now = time.time()
            if self.click_indicator_start and (now - self.click_indicator_start) < self.click_indicator_duration:
                cv2.putText(frame, "Click!", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if self.switch_indicator_start and (now - self.switch_indicator_start) < self.click_indicator_duration:
                cv2.putText(frame, "Switch!", (w//2 - 50, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
            if self.brightness_indicator_start and (now - self.brightness_indicator_start) < self.indicator_duration:
                bri_level = status.get('brightness', 0)
                if bri_level is not None:
                    cv2.rectangle(frame, (10, 10), (int(200 * (bri_level / 100.0)) + 10, 30), (255, 165, 0), -1)
                    cv2.putText(frame, f"Bri: {bri_level}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if self.volume_indicator_start and (now - self.volume_indicator_start) < self.indicator_duration:
                vol_level = status.get('volume', 0)
                if vol_level is not None:
                    cv2.rectangle(frame, (10, 40), (int(200 * (vol_level / 100.0)) + 10, 60), (0, 255, 0), -1)
                    cv2.putText(frame, f"Vol: {vol_level}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            fps_cnt += 1
            if time.time() - fps_t0 >= 1.0:
                current_fps = fps_cnt; fps_cnt = 0; fps_t0 = time.time()

            cv2.putText(frame, f"FPS: {current_fps}  (proc every {self.process_every_n})", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            fs = pyautogui.FAILSAFE
            cv2.putText(frame, f"FAILSAFE: {'ON' if fs else 'OFF'}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if fs else (0,0,255), 2)

            loop_dt = time.time() - t0
            if loop_dt < self.target_dt:
                time.sleep(self.target_dt - loop_dt)
            if self.adaptive_pacing and status['mode'] != 'Mouse':
                self.adaptive_backoff(loop_dt)

            self.frame_signal.emit(frame)
            self.status_signal.emit(status)

        cap.release()
        self.mp.close()

    def stop(self):
        self.run_flag = False
        self.wait()

# -------------------- Main Window --------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.s = load_settings()
        pyautogui.FAILSAFE = bool(self.s.get('failsafe', False))

        global bri_worker, vol_worker
        bri_worker = BrightnessWorker(
            throttle_hz=self.s.get('brightness_throttle_hz', 8),
            ignore_delta=self.s.get('brightness_ignore_delta', 2),
            change_speed=self.s.get('brightness_change_speed', 1.0)
        )
        vol_worker = VolumeWorker(
            throttle_hz=self.s.get('volume_throttle_hz', 8),
            ignore_delta=self.s.get('volume_ignore_delta', 2),
            change_speed=self.s.get('volume_change_speed', 1.0)
        )

        self.setWindowTitle("Smart Control App — Smooth Keyboard Edition")
        self.resize(1240, 840)
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        self.setStyleSheet(self._dark_qss())

        self.tabs = QTabWidget()
        self.video_label = QLabel()
        self.video_label.setFixedSize(960, 720)
        self.status_label = QLabel("Status: Init")
        self.status_label.setFont(QFont("Arial", 12))

        cam_tab = QWidget()
        cam_v = QVBoxLayout()
        cam_v.addWidget(self.video_label)
        cam_v.addWidget(self.status_label)
        cam_tab.setLayout(cam_v)
        self.tabs.addTab(cam_tab, "Camera")

        ctrl_tab = QWidget()
        ctrl_h = QHBoxLayout()
        vol_box = QGroupBox("Volume")
        vb = QVBoxLayout()
        self.vol_slider = QSlider(Qt.Vertical)
        self.vol_slider.setRange(0,100)
        self.vol_slider.setValue(int((sysvol.cur or 0.5)*100))
        self.vol_slider.valueChanged.connect(self.on_volume)
        self.vol_label = QLabel(f"{self.vol_slider.value()}%")
        vb.addWidget(self.vol_slider)
        vb.addWidget(self.vol_label)
        vol_box.setLayout(vb)
        bri_box = QGroupBox("Brightness")
        bb = QVBoxLayout()
        self.bri_slider = QSlider(Qt.Vertical)
        self.bri_slider.setRange(0,100)
        self.bri_slider.setValue(50)
        self.bri_slider.valueChanged.connect(self.on_brightness)
        self.bri_label = QLabel("50%")
        bb.addWidget(self.bri_slider)
        bb.addWidget(self.bri_label)
        bri_box.setLayout(bb)
        ctrl_h.addWidget(vol_box)
        ctrl_h.addWidget(bri_box)
        ctrl_tab.setLayout(ctrl_h)
        self.tabs.addTab(ctrl_tab, "Controls")

        sets = QWidget()
        form = QFormLayout()
        self.chk_help = QCheckBox("Show on-screen help")
        self.chk_help.setChecked(self.s.get('show_help', True))
        self.chk_help.stateChanged.connect(lambda x: self._set('show_help', bool(x)))
        self.chk_mirror = QCheckBox("Mirror camera")
        self.chk_mirror.setChecked(self.s.get('mirror', True))
        self.chk_mirror.stateChanged.connect(lambda x: self._set('mirror', bool(x)))
        self.chk_overlay = QCheckBox("Draw overlay")
        self.chk_overlay.setChecked(self.s.get('draw_overlay', True))
        self.chk_overlay.stateChanged.connect(lambda x: self._set('draw_overlay', bool(x)))
        self.chk_minimal = QCheckBox("Minimal overlay (faster)")
        self.chk_minimal.setChecked(self.s.get('minimal_overlay', True))
        self.chk_minimal.stateChanged.connect(lambda x: self._set('minimal_overlay', bool(x)))
        self.chk_fail = QCheckBox("Enable PyAutoGUI FailSafe (move cursor to corner to stop)")
        self.chk_fail.setChecked(self.s.get('failsafe', False))
        self.chk_fail.stateChanged.connect(self.on_failsafe)
        self.chk_adaptive = QCheckBox("Enable adaptive pacing")
        self.chk_adaptive.setChecked(self.s.get('adaptive_pacing', True))
        self.chk_adaptive.stateChanged.connect(lambda x: self._set('adaptive_pacing', bool(x)))

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.05, 0.95)
        self.spin_alpha.setSingleStep(0.01)
        self.spin_alpha.setValue(self.s.get('mouse_ema_alpha', 0.35))
        self.spin_alpha.valueChanged.connect(lambda v: self._set('mouse_ema_alpha', float(v)))
        self.spin_gate = QDoubleSpinBox()
        self.spin_gate.setRange(0.001, 0.2)
        self.spin_gate.setSingleStep(0.001)
        self.spin_gate.setValue(self.s.get('mouse_vel_gate', 0.015))
        self.spin_gate.valueChanged.connect(lambda v: self._set('mouse_vel_gate', float(v)))

        self.spin_procN = QSpinBox()
        self.spin_procN.setRange(1,4)
        self.spin_procN.setValue(self.s.get('process_every_n', 1))
        self.spin_procN.valueChanged.connect(lambda v: self._set('process_every_n', int(v)))
        self.spin_targetfps = QSpinBox()
        self.spin_targetfps.setRange(15,60)
        self.spin_targetfps.setValue(self.s.get('target_fps', 30))
        self.spin_targetfps.valueChanged.connect(lambda v: self._set('target_fps', int(v)))

        self.spin_bri_speed = QDoubleSpinBox()
        self.spin_bri_speed.setRange(0.1, 2.0)
        self.spin_bri_speed.setSingleStep(0.1)
        self.spin_bri_speed.setValue(self.s.get('brightness_change_speed', 1.0))
        self.spin_bri_speed.valueChanged.connect(self.on_bri_speed_change)
        self.spin_bri_dead = QDoubleSpinBox()
        self.spin_bri_dead.setRange(0.01, 0.1)
        self.spin_bri_dead.setSingleStep(0.01)
        self.spin_bri_dead.setValue(self.s.get('brightness_dead_zone', 0.03))
        self.spin_bri_dead.valueChanged.connect(lambda v: self._set('brightness_dead_zone', float(v)))
        self.spin_vol_speed = QDoubleSpinBox()
        self.spin_vol_speed.setRange(0.1, 2.0)
        self.spin_vol_speed.setSingleStep(0.1)
        self.spin_vol_speed.setValue(self.s.get('volume_change_speed', 1.0))
        self.spin_vol_speed.valueChanged.connect(self.on_vol_speed_change)
        self.spin_vol_dead = QDoubleSpinBox()
        self.spin_vol_dead.setRange(0.01, 0.1)
        self.spin_vol_dead.setSingleStep(0.01)
        self.spin_vol_dead.setValue(self.s.get('volume_dead_zone', 0.03))
        self.spin_vol_dead.valueChanged.connect(lambda v: self._set('volume_dead_zone', float(v)))
        self.spin_mode_lock = QSpinBox()
        self.spin_mode_lock.setRange(100, 1000)
        self.spin_mode_lock.setSingleStep(50)
        self.spin_mode_lock.setValue(self.s.get('control_mode_lock_ms', 500))
        self.spin_mode_lock.valueChanged.connect(lambda v: self._set('control_mode_lock_ms', int(v)))
        self.spin_kb_offset_x = QSpinBox()
        self.spin_kb_offset_x.setRange(-200, 200)
        self.spin_kb_offset_x.setValue(self.s.get('keyboard_offset_x', 0))
        self.spin_kb_offset_x.valueChanged.connect(lambda v: self._set('keyboard_offset_x', int(v)))
        self.spin_kb_offset_y = QSpinBox()
        self.spin_kb_offset_y.setRange(-200, 200)
        self.spin_kb_offset_y.setValue(self.s.get('keyboard_offset_y', 0))
        self.spin_kb_offset_y.valueChanged.connect(lambda v: self._set('keyboard_offset_y', int(v)))
        self.spin_mouse_sens = QDoubleSpinBox()
        self.spin_mouse_sens.setRange(0.5, 3.0)
        self.spin_mouse_sens.setSingleStep(0.1)
        self.spin_mouse_sens.setValue(self.s.get('mouse_sensitivity', 1.0))
        self.spin_mouse_sens.valueChanged.connect(lambda v: self._set('mouse_sensitivity', float(v)))
        self.spin_click_dist = QSpinBox()
        self.spin_click_dist.setRange(0, 100)
        self.spin_click_dist.setSingleStep(5)
        self.spin_click_dist.setValue(self.s.get('click_min_distance', 10))
        self.spin_click_dist.valueChanged.connect(lambda v: self._set('click_min_distance', int(v)))
        self.spin_click_hold = QSpinBox()
        self.spin_click_hold.setRange(50, 500)
        self.spin_click_hold.setSingleStep(10)
        self.spin_click_hold.setValue(self.s.get('click_hold_time_ms', 100))
        self.spin_click_hold.valueChanged.connect(lambda v: self._set('click_hold_time_ms', int(v)))
        self.spin_click_smooth = QDoubleSpinBox()
        self.spin_click_smooth.setRange(0.1, 0.9)
        self.spin_click_smooth.setSingleStep(0.05)
        self.spin_click_smooth.setValue(self.s.get('click_smoothing_alpha', 0.4))
        self.spin_click_smooth.valueChanged.connect(lambda v: self._set('click_smoothing_alpha', float(v)))
        self.spin_click_finger_dist = QDoubleSpinBox()
        self.spin_click_finger_dist.setRange(0.02, 0.1)
        self.spin_click_finger_dist.setSingleStep(0.01)
        self.spin_click_finger_dist.setValue(self.s.get('click_finger_distance', 0.04))
        self.spin_click_finger_dist.valueChanged.connect(lambda v: self._set('click_finger_distance', float(v)))
        self.spin_hand_switch_dist = QDoubleSpinBox()
        self.spin_hand_switch_dist.setRange(0.02, 0.1)
        self.spin_hand_switch_dist.setSingleStep(0.01)
        self.spin_hand_switch_dist.setValue(self.s.get('hand_switch_distance', 0.05))
        self.spin_hand_switch_dist.valueChanged.connect(lambda v: self._set('hand_switch_distance', float(v)))

        form.addRow(self.chk_help)
        form.addRow(self.chk_mirror)
        form.addRow(self.chk_overlay)
        form.addRow(self.chk_minimal)
        form.addRow("Mouse smoothing alpha:", self.spin_alpha)
        form.addRow("Velocity gate:", self.spin_gate)
        form.addRow("Process every N frames:", self.spin_procN)
        form.addRow("Target FPS:", self.spin_targetfps)
        form.addRow("Brightness change speed:", self.spin_bri_speed)
        form.addRow("Brightness dead zone:", self.spin_bri_dead)
        form.addRow("Volume change speed:", self.spin_vol_speed)
        form.addRow("Volume dead zone:", self.spin_vol_dead)
        form.addRow("Control mode lock (ms):", self.spin_mode_lock)
        form.addRow("Keyboard offset X:", self.spin_kb_offset_x)
        form.addRow("Keyboard offset Y:", self.spin_kb_offset_y)
        form.addRow("Mouse sensitivity (DPI):", self.spin_mouse_sens)
        form.addRow("Minimum click distance (px):", self.spin_click_dist)
        form.addRow("Click hold time (ms):", self.spin_click_hold)
        form.addRow("Click smoothing alpha:", self.spin_click_smooth)
        form.addRow("Click finger distance:", self.spin_click_finger_dist)
        form.addRow("Hand switch distance:", self.spin_hand_switch_dist)
        form.addRow(self.chk_adaptive)
        form.addRow(self.chk_fail)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save Settings")
        btn_save.clicked.connect(lambda: (save_settings(self.s), QMessageBox.information(self, "Settings", "Saved")))
        btn_apply = QPushButton("Apply Changes")
        btn_apply.clicked.connect(self.apply_changes)
        btn_reset = QPushButton("Reset Defaults")
        btn_reset.clicked.connect(self.reset_defaults)
        btn_calib = QPushButton("Calibration Wizard")
        btn_calib.clicked.connect(self.run_calibration)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_calib)
        form.addRow(btn_row)

        sets.setLayout(form)
        self.tabs.addTab(sets, "Settings")

        root = QVBoxLayout()
        root.addWidget(self.tabs)
        self.setLayout(root)

        self.th = VideoThread(self.s)
        self.th.frame_signal.connect(self.update_image)
        self.th.status_signal.connect(self.update_status)
        self.th.start()

        self.perf = QLabel("")
        self.perf.setFont(QFont("Arial", 10))
        cam_v.addWidget(self.perf)
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.tick_perf)
        self.perf_timer.start(1000)

    def _set(self, k, v):
        self.s[k] = v
        save_settings(self.s)

    def on_failsafe(self, st):
        val = bool(st)
        self.s['failsafe'] = val
        pyautogui.FAILSAFE = val
        save_settings(self.s)

    def on_volume(self, val):
        self.vol_label.setText(f"{val}%")
        sysvol.set_norm(val/100.0)

    def on_brightness(self, val):
        self.bri_label.setText(f"{val}%")
        bri_worker.set(val)

    def on_bri_speed_change(self, v):
        self._set('brightness_change_speed', float(v))
        global bri_worker
        bri_worker.change_speed = float(v)

    def on_vol_speed_change(self, v):
        self._set('volume_change_speed', float(v))
        global vol_worker
        vol_worker.change_speed = float(v)

    def apply_changes(self):
        save_settings(self.s)
        self.th.stop()
        global bri_worker, vol_worker
        bri_worker.stop()
        vol_worker.stop()
        bri_worker = BrightnessWorker(
            throttle_hz=self.s.get('brightness_throttle_hz', 8),
            ignore_delta=self.s.get('brightness_ignore_delta', 2),
            change_speed=self.s.get('brightness_change_speed', 1.0)
        )
        vol_worker = VolumeWorker(
            throttle_hz=self.s.get('volume_throttle_hz', 8),
            ignore_delta=self.s.get('volume_ignore_delta', 2),
            change_speed=self.s.get('volume_change_speed', 1.0)
        )
        self.th = VideoThread(self.s)
        self.th.frame_signal.connect(self.update_image)
        self.th.status_signal.connect(self.update_status)
        self.th.start()
        QMessageBox.information(self, "Settings", "Changes applied.")

    def reset_defaults(self):
        self.s.clear()
        self.s.update(DEFAULTS.copy())
        save_settings(self.s)
        QMessageBox.information(self, "Settings", "Defaults restored. Apply changes to take effect.")

    def run_calibration(self):
        instructions = [
            ("top-left", (0, 0)),
            ("top-right", (self.th.screen_w - 1, 0)),
            ("bottom-left", (0, self.th.screen_h - 1)),
            ("bottom-right", (self.th.screen_w - 1, self.th.screen_h - 1))
        ]
        src_points = []
        for corner, dst_pt in instructions:
            msg = f"Point your index finger on your RIGHT hand to the {corner.upper()} corner of your screen. Hold steady, then click OK to capture."
            reply = QMessageBox.information(self, "Calibration", msg, QMessageBox.Ok | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return
            pos = self.th.current_idx_pos
            if pos is None:
                QMessageBox.warning(self, "Calibration Error", "No right hand detected. Try again.")
                return
            src_points.append(pos)
        if len(src_points) == 4:
            src = np.array(src_points, dtype=np.float32)
            dst = np.array([inst[1] for inst in instructions], dtype=np.float32)
            H, _ = cv2.findHomography(src, dst)
            self.s['calib_A'] = H.tolist()
            save_settings(self.s)
            QMessageBox.information(self, "Calibration", "Calibration complete. Apply changes to take effect.")
        else:
            QMessageBox.warning(self, "Calibration Error", "Failed to collect all points.")

    def tick_perf(self):
        txt = f"GPU: CPU (no CUDA)"
        self.perf.setText(txt)

    def update_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self, st):
        vol = st.get('volume'); bri = st.get('brightness'); mode = st.get('mode'); fps = st.get('fps')
        vtxt = f"{vol}%" if isinstance(vol, int) else "N/A"
        btxt = f"{bri}%" if isinstance(bri, int) else "N/A"
        typed = st.get('typed')
        extra = f" | Typed: {typed}" if typed else ""
        self.status_label.setText(f"Mode: {mode} | FPS: {fps} | Vol: {vtxt} | Br: {btxt}{extra}")

        if mode == 'Control':
            self.vol_slider.setEnabled(False)
            self.bri_slider.setEnabled(False)
        else:
            self.vol_slider.setEnabled(True)
            self.bri_slider.setEnabled(True)

        if vol is not None:
            self.vol_slider.blockSignals(True)
            self.vol_slider.setValue(vol)
            self.vol_slider.blockSignals(False)
            self.vol_label.setText(f"{vol}%")
        if bri is not None:
            self.bri_slider.blockSignals(True)
            self.bri_slider.setValue(bri)
            self.bri_slider.blockSignals(False)
            self.bri_label.setText(f"{bri}%")

    def closeEvent(self, e):
        save_settings(self.s)
        try:
            self.th.stop()
        except Exception:
            pass
        try:
            bri_worker.stop()
        except Exception:
            pass
        try:
            vol_worker.stop()
        except Exception:
            pass
        e.accept()

    def _dark_qss(self):
        return """
        QWidget { background: #121417; color: #eaeaea; }
        QTabWidget::pane { border: 1px solid #333; }
        QGroupBox { border: 1px solid #333; border-radius: 8px; margin-top: 12px; padding: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QLabel { font-size: 14px; }
        QPushButton { background: #1f2329; border: 1px solid #444; border-radius: 8px; padding: 8px 14px; }
        QPushButton:hover { background: #2a2f36; }
        QSlider::groove:vertical { background: #2a2f36; width: 6px; border-radius: 3px; }
        QSlider::handle:vertical { height: 16px; background: #4b8cf7; margin: 0 -4px; border-radius: 6px; }
        QCheckBox { font-size: 14px; }
        """

# -------------------- Entry --------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())