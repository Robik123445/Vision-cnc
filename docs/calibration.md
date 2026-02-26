# Vision Calibration (Intrinsics + CNC Plane)

## 1) Intrinsic calibration
1. Print chessboard (default `9x6`) and place on CNC table at different angles/distances.
2. Capture at least 15-25 sharp frames.
3. Run:
   ```bash
   python scripts/vision_calibrate_intrinsics.py --images_dir calibration/chessboard --pattern 9x6 --square_mm 25.0 --out calibration/intrinsics.json
   ```
4. Output file: `calibration/intrinsics.json`.

Tip: použite rôzne pozície a uhly, aby bol distortion model stabilný.

## 2) Plane calibration (pixel -> mm)
1. Prepare one top-view frame of the CNC workspace.
2. Run:
   ```bash
   python scripts/vision_calibrate_plane_manual.py --image calibration/plane_frame.jpg --workspace_mm 800x500 --intrinsics calibration/intrinsics.json --out calibration/plane.json
   ```
3. Click 4 corners in order: `(0,0)`, `(W,0)`, `(W,H)`, `(0,H)`.
4. Press `ENTER` to save.

## 3) Verification with mm grid overlay
Use smoketest with projected mm grid:
```bash
python scripts/vision_detect_smoketest.py --image calibration/plane_frame.jpg --with_mm_grid
```
`runs/<timestamp>/preview.png` should show aligned grid on workspace edges.

## Runtime API
- `undistort_frame(frame, intrinsics)`
- `pixel_to_mm(x_px, y_px, plane_calib)`
- `mask_px_to_mask_mm_grid(mask_px, calib, target_shape, workspace_mm)`
