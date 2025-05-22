import numpy as np
import cv2

def load_calib(calib_file):
    calib = {}
    with open(calib_file) as f:
        for line in f:
            key, *vals = line.strip().split()
            calib[key] = np.array(vals, dtype=np.float32).reshape(-1)
    return calib

def doColorize(pc_file, im_file, calib_file):
    # 1. Load point cloud
    points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4) # (x, y, z, intensity)
    pts_velo = points[:, :3] # (x, y, z)
    # 2. Load image
    img = cv2.imread(im_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    # 3. Load calibration
    calib = load_calib(calib_file)
    P2 = calib['P2:'].reshape(3, 4)  # Left camera proj
    Tr = calib['Tr:'].reshape(3, 4) # LiDAR to cam0
    # 4. Transform LiDAR points to camera frame
    pts_hom = np.hstack((pts_velo, np.ones((pts_velo.shape[0], 1))))  # N×4
    pts_cam = ((Tr @ pts_hom.T).reshape(3, -1)).T  # N×3 
    # 5. Project to image plane
    pts_img = (P2 @ np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1)))).T).T
    pts_img[:, 0] /= pts_img[:, 2]
    pts_img[:, 1] /= pts_img[:, 2]
    # 6. Filter valid points
    valid = (pts_cam[:, 2] > 0) & \
            (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & \
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
    pts_valid = pts_velo[valid]
    uv = pts_img[valid, :2].astype(int)
    colors = img[uv[:, 1], uv[:, 0]] / 255.0  # RGB normalized
    # 7. Print number of valid points along with number of points
    print(f'Number of valid points/ Number of points: {np.sum(valid)}/{pts_velo.shape[0]} ({np.sum(valid)/pts_velo.shape[0]*100:.2f}%)')
    # 8. Cascade the valid array to points
    points_add = np.concatenate((points, np.zeros((points.shape[0], 4))), axis=1)
    points_add[:, 4] = valid.astype(np.int32)  # Assign 1 if valid, 0 otherwise 
    points_add[valid, 5:8] = colors  # RGB
    return points_add, pts_valid, colors