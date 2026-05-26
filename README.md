# KITTI LiDAR-Camera Projection

Projects Velodyne LiDAR point clouds onto camera images using the KITTI dataset calibration files, and renders the results as a video.

---

## What It Does

1. Loads raw LiDAR scans (`.bin`) and camera images from a KITTI drive sequence
2. Applies the `calib_velo_to_cam.txt` extrinsic transform to convert LiDAR points into the camera frame
3. Applies the `calib_cam_to_cam.txt` intrinsic projection matrix to map 3D points onto the 2D image plane
4. Filters out points behind the camera and outside image bounds
5. Colours each projected point by depth using the `turbo` colormap
6. Renders all frames into an `.mp4` video

---

## Project Structure

```
.
├── lidar_projection.py                        # Main script
├── lidar_projection.mp4            # Output video (generated)
└── 2011_09_26_drive_0005_sync/
    └── 2011_09_26/
        ├── calib_velo_to_cam.txt   # LiDAR → camera extrinsics
        ├── calib_cam_to_cam.txt    # Camera intrinsics
        └── 2011_09_26_drive_0005_sync/
            ├── image_02/data/      # Camera frames (.png)
            └── velodyne_points/data/ # LiDAR scans (.bin)
```

---

## Requirements

```bash
pip install numpy matplotlib opencv-python Pillow
```

| Package | Purpose |
|---|---|
| `numpy` | Point cloud math and matrix operations |
| `matplotlib` | Rendering projected points onto images |
| `opencv-python` | Writing frames to `.mp4` |
| `Pillow` | Image loading |

---

## Dataset

Download from the [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) page.

You will need:
- A **synced+rectified** drive sequence (e.g. `2011_09_26_drive_0005_sync`)
- The **calibration files** for that date (`calib_velo_to_cam.txt`, `calib_cam_to_cam.txt`)

---

## Usage

1. Update the `BASE` path at the top of `video.py` to point to your drive sequence:

```python
BASE = Path('2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync')
```

2. Run the script:

```bash
python lidar_projection.py
```

3. Output is saved as `lidar_projection.mp4` in the working directory.

---

## Configuration

All tuneable parameters are at the top of `lidar_projection.py`:

| Variable | Default | Description |
|---|---|---|
| `FPS` | `10` | Output video framerate (KITTI is captured at 10 Hz) |
| `IMG_WIDTH` | `1242` | KITTI image width in pixels |
| `IMG_HEIGHT` | `375` | KITTI image height in pixels |
| `OUTPUT_VIDEO` | `lidar_projection.mp4` | Output file path |

---

## How the Projection Works

LiDAR points are projected onto the image plane in two steps:

```
LiDAR (x,y,z) → [T_velo_cam] → Camera 3D → [P_rect_02] → Image (u,v)
```

These are combined into a single 3×4 matrix to avoid redundant computation:

```python
P = P_rect_02 @ T_velo_cam   # applied once per frame
```

Points are filtered at two stages:
- **Depth > 0** — removes points behind the camera
- **Bounds check** — removes points outside the image dimensions

---

## References

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
