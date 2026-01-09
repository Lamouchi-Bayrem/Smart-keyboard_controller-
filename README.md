# Smart Control App - Smooth Keyboard Edition

## Overview

This is a gesture-based control application that uses computer vision to detect hand gestures for controlling your computer. It supports mouse movement, clicking, dragging, volume and brightness adjustment, and an on-screen keyboard for typing. The app is built using Python with libraries like OpenCV, MediaPipe, NumPy, PyAutoGUI, and PyQt5 for the UI.

Key features:
- **Mouse Control**: Use your right hand to move the cursor, pinch for drag, and bring index and middle fingers close for left-click.
- **Control Mode**: Left hand for adjusting system volume or brightness based on gesture orientation.
- **Keyboard Mode**: When two hands are detected, an on-screen QWERTY keyboard appears. Hover over keys with the active hand's index finger to type. Switch active hand by bringing index fingers close.
- **Non-blocking Workers**: Smooth brightness and volume changes using background threads.
- **Calibration**: Perspective calibration wizard for accurate mouse mapping.
- **Configurable Settings**: Adjustable parameters for smoothing, thresholds, offsets, and more, saved in JSON.

## Requirements

- Python 3.8+
- Required libraries: `opencv-python`, `mediapipe`, `numpy`, `pyautogui`, `PyQt5`
- Optional: `screen_brightness_control` for better brightness control, `pycaw` for Windows audio control

Install dependencies:
```
pip install opencv-python mediapipe numpy pyautogui PyQt5 screen_brightness_control pycaw
```

Note: For non-Windows OS, volume control may fall back to keyboard shortcuts, which could be less precise.

## Usage

1. Run the script:
   ```
   python smartcontroller.py
   ```

2. The app window opens with tabs for Camera (live feed), Controls (manual sliders), and Settings.

3. **Gestures**:
   - **Mouse Mode** (Right hand only): Index finger for cursor, index + middle close for click, thumb + index pinch for drag.
   - **Control Mode** (Left hand only): Thumb to index distance controls value; vertical orientation for brightness, horizontal for volume.
   - **Keyboard Mode** (Two hands): Active index finger hovers over keys to type; switch hands by touching index fingers.

4. **Settings**: Customize smoothing, thresholds, FPS, offsets, etc. Click "Apply Changes" to reload without restarting the window.

5. **Calibration**: Use the "Calibration Wizard" in Settings to map hand positions to full screen for better mouse control.

## Configuration

Settings are saved in `multigame_settings.json`. Key parameters:
- `mouse_ema_alpha`: Cursor smoothing (0.05-0.95)
- `control_ema_alpha`: Brightness/volume smoothing (0.05-0.95)
- `click_finger_distance`: Index-middle distance for click (0.02-0.1)
- `hand_switch_distance`: Index fingers distance for keyboard hand switch (0.02-0.1)
- `brightness_dead_zone` / `volume_dead_zone`: Ignore small changes (0.01-0.1)
- `control_mode_lock_ms`: Lock control mode duration (100-1000ms)

## Troubleshooting

- **Camera Issues**: Check `camera_index` in settings.
- **Volume/Brightness Not Working**: Ensure optional libraries are installed; fallback uses keyboard shortcuts.
- **Jittery Controls**: Increase smoothing alphas or dead zones in settings.
- **FPS Low**: Reduce `target_fps` or enable `minimal_overlay`.

## Limitations

- Works best with good lighting and clear hand visibility.
- Volume control may vary by OS; no direct support for all backends.
- No multi-monitor support yet.
- Keyboard is basic QWERTY; no symbols/numbers.
-For the next step the code will be organized into init.py and we will add a launch file named run.py to automate the checks of dependencys and troubleshooting 

## Contributing

Feel free to fork and submit PRs for improvements, e.g., more gestures, better OS compatibility, or extended keyboard layouts.

## License

MIT License. See LICENSE file for details.
