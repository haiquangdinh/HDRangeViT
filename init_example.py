import numpy as np
import cv2
import open3d as o3d

# 0. Set paths: cd to the directory ../SemanticKITTI/dataset/sequences/00
basedir = '../SemanticKITTI/dataset/sequences/00'

# 1. Load point cloud
file = f'{basedir}/velodyne/000000.bin'
points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
pts_velo = points[:, :3]

# 2. Load image
file = f'{basedir}/image_2/000000.png'
img = cv2.imread(file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# 3. Load calibration
def load_calib(path):
    calib = {}
    with open(path) as f:
        for line in f:
            key, *vals = line.strip().split()
            calib[key] = np.array(vals, dtype=np.float32).reshape(-1)
    return calib

file = f'{basedir}/calib.txt'
calib = load_calib(file)

P2 = calib['P2:'].reshape(3, 4)  # Left camera projection matrix
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

# print number of valid points along with number of points
print(f'Number of valid points/ Number of points: {np.sum(valid)}/{pts_velo.shape[0]} ({np.sum(valid)/pts_velo.shape[0]*100:.2f}%)')

# 7. Visualize with Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_valid)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

# save the visualizatio to jpeg file
# o3d.io.write_point_cloud('pointcloud.ply', pcd)
# 8. Visualize with OpenCV
# img = cv2.imread(file)
# for i in range(pts_valid.shape[0]):
#     cv2.circle(img, (uv[i, 0], uv[i, 1]), 1, (0, 255, 0), -1)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import pykitti

# basedir = '../SemanticKITTI/dataset/sequences/00'
# date = '2011_09_26'
# drive = '0019'

# # The 'frames' argument is optional - default: None, which loads the whole dataset.
# # Calibration, timestamps, and IMU data are read automatically. 
# # Camera and velodyne data are available via properties that create generators
# # when accessed, or through getter methods that provide random access.
# data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))

# # dataset.calib:         Calibration data are accessible as a named tuple
# # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# # dataset.camN:          Returns a generator that loads individual images from camera N
# # dataset.get_camN(idx): Returns the image from camera N at idx  
# # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# # dataset.get_gray(idx): Returns the monochrome stereo pair at idx  
# # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx  
# # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# # dataset.get_velo(idx): Returns the velodyne scan at idx  

# point_velo = np.array([0,0,0,1])
# point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

# point_imu = np.array([0,0,0,1])
# point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]

# for cam0_image in data.cam0:
#     # do something
#     pass

# cam2_image, cam3_image = data.get_rgb(3)

# import numpy as np
# points = np.fromfile('../SemanticKITTI/dataset/sequences/00/velodyne/000000.bin', dtype=np.float32).reshape(-1, 4)
# # show the shape of points
# print(points.shape)
# # show the first 5 points
# print(points[:5])
# # show the range of each column
# print(np.min(points, axis=0))
# print(np.max(points, axis=0))

