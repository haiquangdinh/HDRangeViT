### Projection
import numpy as np
class ScanProjection(object):
    '''
    Project the 3D point cloud to 2D data with range projection

    Adapted from A. Milioto et al. https://github.com/PRBonn/lidar-bonnetal
    '''

    def __init__(self, proj_w, proj_h):
        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h


    def doProjection(self, pointcloud: np.ndarray):

        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
        # label is the last column of pointcloud
        label = pointcloud[:,-1]
        # remove the last column from pointcloud
        pointcloud = pointcloud[:, :-1]
        # remove flag, R, G, and B from pointcloud
        pointcloud = pointcloud[:, :-4]  # now only has [x, y, z, intensity]
        # get angles of all points
        yaw = -np.arctan2(y, -x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        #breakpoint()
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        proj_y = np.cumsum(proj_y)
        # scale to image size using angular resolution
        proj_x = proj_x * self.proj_w - 0.001

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)
        # wrap proj_x so if proj_x < 1024 it will be added 1024, if proj_x >= 1024 it will be subtracted 1024
        proj_x = np.where(proj_x < 1024, proj_x + 1024, proj_x - 1024)

        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        label = label[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full((self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_label = np.full((self.proj_h, self.proj_w), 0, dtype=np.int32)
        proj_label[proj_y, proj_x] = label

        # create proj_tensor with cascade proj_pointcloud and proj_range
        # proj_pointcloud has size (64, 2048, 4)
        # proj_range has size (64, 2048)
        proj_pointcloud = proj_pointcloud[0:48, 790:1250, :]  # cut the pointcloud to [H, W, C]
        proj_range =proj_range[0:48, 790:1250]  # cut the range to [H, W]
        proj_label = proj_label[0:48, 790:1250]

        proj_tensor = np.concatenate((proj_range[..., np.newaxis], proj_pointcloud), axis=-1) # [range, x, y, z, flag, R, G, B]
        return proj_tensor, proj_label