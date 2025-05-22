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

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray, wrap_around: int = 0):
        self.cached_data = {}
        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

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
        # --- Wrap around vertical axis if specified ---
        if wrap_around > 0 and wrap_around < self.proj_w:
            # Shift proj_x so that wrap_around is the new "zero"
            proj_x = (proj_x - wrap_around) % self.proj_w
            
        # print(f'proj_y: [{proj_y.min()} - {proj_y.max()}] - ({(proj_y < self.proj_h).astype(np.int32).sum()} - {(proj_y >= self.proj_h).astype(np.int32).sum()})')

        px = proj_x.copy()
        py = proj_y.copy()

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)

        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        self.cached_data['uproj_x_idx'] = proj_x.copy()
        self.cached_data['uproj_y_idx'] = proj_y.copy()
        self.cached_data['uproj_depth'] = depth.copy()
        self.cached_data['px'] = px
        self.cached_data['py'] = py

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask
