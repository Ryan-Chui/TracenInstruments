# Camera and projector calibration

This folder contains the calibration scripts and the printable
[ChArUco target](./charuco.pdf). The reusable calibration library remains in
`scanner/calibration/`.

## Recalibrate the ELP camera

Use the exact capture mode that will be used for scanning. For example, from
the repository root:

```powershell
python .\calibration\1-calibrate_camera.py --camera 0 --width 1920 --height 1080 --fps 20
```

If the ELP appears at another device index, try `--camera 1`, `--camera 2`,
and so on. The script opens a preview:

- `c`: save a ChArUco image
- `d`: toggle undistorted preview
- `k`: calibrate and save `cam_mtx.npy` and `cam_dist.npy`
- `q`: quit

Capture at least 15 sharp images with the board filling different parts of the
frame and tilted at different angles. Keep the board flat and make sure all
printed markers are visible. Do not mix resolutions, crops, autofocus states,
or aspect ratios in one calibration set.

The default ELP outputs are:

```text
data/CalibrationImgs/elp_camera/
data/calib_results/elp_camera/
```

To use a different capture mode or preserve separate datasets:

```powershell
python .\calibration\1-calibrate_camera.py `
  --camera 1 --width 1280 --height 720 --fps 30 `
  --images .\data\CalibrationImgs\elp_1280x720 `
  --results .\data\calib_results\elp_1280x720
```

The printed reprojection error is useful for comparing calibration attempts,
but a low number does not compensate for blurry or poorly distributed images.

## Calibrate the projector-camera pair

After camera calibration, update the camera settings in
`2-calibrate_projcam.py` to the same ELP width, height, FPS, and camera index.
Run:

```powershell
python .\calibration\2-calibrate_projcam.py
```

The projector must be connected as an extended display. Follow the preview
instructions, move the ChArUco/white calibration surface through the projected
3x3 grid, and keep the setup rigid. The script saves projector and
camera-projector results under `data/calib_results/`.

If projector photos already exist and only the camera calibration changed, use
`2b-recalibrate.py` after setting `cam_result_folder` to the new ELP results.
This reuses the saved projector observations without repeating capture.

Calibration results are resolution-specific. Recalibrate the camera whenever
the ELP resolution, crop, lens, focus, or camera position changes.
