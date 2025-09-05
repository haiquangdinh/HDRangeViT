# colorize waymo point cloud
import numpy as np
from waymo_open_dataset.utils import range_image_utils
import tensorflow as tf

WAYMO_TO_SEMKITTI_20 = {
    0: 0,   # UNKNOWN -> unlabeled
    1: 1,   # CAR -> car
    2: 4,   # TRUCK -> truck
    3: 4,   # BUS -> truck
    4: 5,   # OTHER_VEHICLE -> other-vehicle
    5: 8,   # MOTORCYCLIST -> motorcyclist
    6: 7,   # BICYCLIST -> bicyclist
    7: 6,   # PEDESTRIAN -> person
    8: 19,  # SIGN -> traffic-sign
    9: 19,  # TRAFFIC_LIGHT -> traffic-sign
    10: 18, # POLE -> pole
    11: 18, # CONSTRUCTION_CONE -> pole (approx)
    12: 2,  # BICYCLE -> bicycle
    13: 3,  # MOTORCYCLE -> motorcycle
    14: 13, # BUILDING -> building
    15: 15, # VEGETATION -> vegetation
    16: 16, # TREE_TRUNK -> trunk
    17: 12, # CURB -> other-ground (alt: road=9)
    18: 9,  # ROAD -> road
    19: 9,  # LANE_MARKER -> road
    20: 12, # OTHER_GROUND -> other-ground
    21: 11, # WALKABLE -> sidewalk
    22: 11, # SIDEWALK -> sidewalk
}


def _sample_rgb_nearest(img, rows, cols):
    h, w = img.shape[:2]
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)
    return img[rows, cols, :]

def colorize_range_image_v2(
    proj,                 # [H, W, 6] = [cam1,row1,col1, cam2,row2,col2]
    camera_images,        # dict: {camera_id:int -> np.uint8 [Hc,Wc,3]}
    default_color=(128,128,128)
):
    """
    Returns:
      colors_hw3: uint8 [H, W, 3]
      used_mask:  bool  [H, W]  (True where a camera projection was used)
    """
    H, W, C = proj.shape
    assert C == 6, "Expected v2 projection with two triplets: [cam1,row1,col1, cam2,row2,col2]"
    colors = np.empty((H, W, 3), dtype=np.uint8)
    colors[:] = np.array(default_color, dtype=np.uint8)
    used = np.zeros((H, W), dtype=bool)

    cam1 = proj[..., 0]; r1 = proj[..., 1]; c1 = proj[..., 2]
    cam2 = proj[..., 3]; r2 = proj[..., 4]; c2 = proj[..., 5]
    # SWAP row and col here!
    r1, c1 = c1, r1
    r2, c2 = c2, r2
    # Pass 1: try primary projection per camera
    for cid, img in camera_images.items():
        m1 = (cam1 == cid)
        if not np.any(m1):
            continue
        rr = r1[m1].astype(np.int32)
        cc = c1[m1].astype(np.int32)
        rgb = _sample_rgb_nearest(img, rr, cc)
        ii, jj = np.nonzero(m1)
        colors[ii, jj, :] = rgb
        used[ii, jj] = True

    # Pass 2: fallback to secondary where primary failed
    need = ~used
    for cid, img in camera_images.items():
        m2 = (cam2 == cid) & need
        if not np.any(m2):
            continue
        rr = r2[m2].astype(np.int32)
        cc = c2[m2].astype(np.int32)
        rgb = _sample_rgb_nearest(img, rr, cc)
        ii, jj = np.nonzero(m2)
        colors[ii, jj, :] = rgb
        used[ii, jj] = True

    return colors, used


def colorize_waymo(label_channel, lidar_channel, extrinsic, beam_inclination, lidar_camera_projection, cam_images):
    # Get the semantic ID (ignore instance ID)
    semantic_id = label_channel[:,:,1]
    # Convert orig_sem_label to sem_label using the map
    sem_label = np.vectorize(WAYMO_TO_SEMKITTI_20.get)(semantic_id)
    # Get x, y, z from lidar_channel
    lidar_range = lidar_channel[:,:,0]
    lidar_intensity = lidar_channel[:,:,1]

    # #########################################################
    # convert lidar_range, extrinsic, and beam_inclination to tensors with [B, W, H] with B=1
    lidar_range = tf.convert_to_tensor(lidar_range, dtype=tf.float32)
    lidar_range = tf.expand_dims(lidar_range, axis=0)
    extrinsic = tf.convert_to_tensor(extrinsic, dtype=tf.float32)
    extrinsic = tf.expand_dims(extrinsic, axis=0)
    beam_inclination = tf.convert_to_tensor(beam_inclination, dtype=tf.float32)
    beam_inclination = tf.expand_dims(beam_inclination, axis=0)
    # Convert to 3D points (vehicle frame)
    points = range_image_utils.extract_point_cloud_from_range_image(lidar_range, extrinsic, beam_inclination)
    #  make point to by numarray
    points_np = points.numpy()
    # squeeze to remove dimensions of size 1
    points_np = points_np.squeeze()
    # sanity check points_np: if lidar_range = -1, then points_np = -1
    points_np[lidar_range == -1] = 0
    #  put the range image so that it has size {H, W} with five channel: range, x, y, z, intensity in such order
    # lidar_raw[:,:,0] then points_np then lidar_raw[:,:,1]
    points_final = np.zeros((64, 2650, 5))
    points_final[:,:,0] = lidar_range
    points_final[:,:,1:4] = points_np
    points_final[:,:,4] = lidar_intensity
    # now we have data that similar to that of the paper. Just need to add R,G, B (which is huge thing to do)
    rgb_grid, used_mask = colorize_range_image_v2(lidar_camera_projection, cam_images, default_color=(0,0,0))
    # combine the final points with range, x, y, z, intensity, flag, R, G, B and label
    # flag is valid when and only when range > 0 and used_mask is one
    points_final_output = np.zeros((64, 2650, 10))
    points_final_output[:,:,0] = points_final[:,:,0]  # range
    points_final_output[:,:,1] = points_final[:,:,1]  # x
    points_final_output[:,:,2] = points_final[:,:,2]  # y
    points_final_output[:,:,3] = points_final[:,:,3]  # z
    points_final_output[:,:,4] = points_final[:,:,4]  # intensity
    points_final_output[:,:,5] = (points_final[:,:,0] > 0) & (used_mask)  # flag
    points_final_output[:,:,6] = rgb_grid[:,:, 0]  # R
    points_final_output[:,:,7] = rgb_grid[:,:, 1]  # G
    points_final_output[:,:,8] = rgb_grid[:,:, 2]  # B
    points_final_output[:,:,9] = sem_label  # label (not used)

    # sanity put: any -1 element is set to 0
    points_final_output[points_final_output == -1] = 0

    return points_final_output  # [H, W, 10]: range, x, y, z, intensity, flag, R, G, B, label