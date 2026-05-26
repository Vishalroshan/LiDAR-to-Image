import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# CONFIG — update these paths
# ─────────────────────────────────────────────
BASE = Path('2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync')

IMG_FOLDER = BASE / 'image_02/data'
LIDAR_FOLDER = BASE / 'velodyne_points/data'
CALIB_VELO_CAM = BASE / '2011_09_26/calib_velo_to_cam.txt'
CALIB_CAM_CAM = BASE / '2011_09_26/calib_cam_to_cam.txt'

OUTPUT_VIDEO = 'lidar_projection.mp4'
FPS = 10        # KITTI is 10 Hz → real-time playback
IMG_WIDTH = 1242
IMG_HEIGHT = 375

# ─────────────────────────────────────────────
# CALIBRATION HELPERS
# ─────────────────────────────────────────────


def load_velo_to_cam(path):
    """Returns 4x4 rigid-body transform: LiDAR → camera frame."""
    data = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    R = np.array(data['R'].split(), dtype=np.float64).reshape(3, 3)
    t = np.array(data['T'].split(), dtype=np.float64)
    g = np.eye(4)
    g[:3, :3] = R
    g[:3,  3] = t
    return g


def load_intrinsics(path):
    """Returns 3x4 projection matrix P_rect_02."""
    data = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    P = np.array(data['P_rect_02'].split(), dtype=np.float64).reshape(3, 4)
    return P


# ─────────────────────────────────────────────
# LIDAR HELPERS
# ─────────────────────────────────────────────
def load_lidar(path):
    """Load a KITTI .bin scan → (N, 3) xyz array."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def project_lidar(pts_xyz, P):
    """
    Project (N,3) LiDAR points through combined 3x4 matrix P = P_rect @ T_velo_cam.
    Returns (M,3) array of [u*z, v*z, z] for points that pass image-bounds filtering.
    """
    N = len(pts_xyz)
    pts_h = np.column_stack([pts_xyz, np.ones(N)])   # (N, 4)
    proj = (P @ pts_h.T).T                            # (N, 3)

    mask = (
        (proj[:, 2] > 0) &
        (proj[:, 0] >= 0) & (proj[:, 0] < IMG_WIDTH * proj[:, 2]) &
        (proj[:, 1] >= 0) & (proj[:, 1] < IMG_HEIGHT * proj[:, 2])
    )
    return proj[mask]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load calibration
    T_velo_cam = load_velo_to_cam(CALIB_VELO_CAM)
    P_rect = load_intrinsics(CALIB_CAM_CAM)
    P = P_rect @ T_velo_cam          # combined 3x4 projection matrix

    # Collect sorted file lists
    img_files = sorted(IMG_FOLDER.iterdir(),   key=lambda p: p.name)
    lidar_files = sorted(LIDAR_FOLDER.iterdir(), key=lambda p: p.name)
    n_frames = min(len(img_files), len(lidar_files))
    print(f"Found {n_frames} frames.")

    # Set up video writer
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    for i, (img_path, lidar_path) in enumerate(zip(img_files, lidar_files)):
        pts_xyz = load_lidar(lidar_path)
        proj = project_lidar(pts_xyz, P)

        depth = proj[:, 2]
        u = (proj[:, 0] / depth).astype(int)
        v = (proj[:, 1] / depth).astype(int)

        img = mpimg.imread(img_path)

        fig, ax = plt.subplots(
            figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100), dpi=100)
        ax.imshow(img)
        ax.scatter(u, v, c=depth, cmap='turbo', s=1, alpha=0.3)
        ax.set_xlim(0, IMG_WIDTH)
        ax.set_ylim(IMG_HEIGHT, 0)
        ax.axis('off')
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_frames} frames done...")

    out.release()
    print(f"\nSaved → {OUTPUT_VIDEO}")


if __name__ == '__main__':
    main()
