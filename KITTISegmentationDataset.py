### DataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ScanProjection import ScanProjection
class KITTISegmentationDataset(Dataset):
    def __init__(self, root_dir, sequences):
        self.root_dir = root_dir
        self.file_list = []
        for seq in sequences:
            seq_dir = os.path.join(root_dir, seq)
            assert os.path.exists(seq_dir), f"Sequence {seq} does not exist in {root_dir}"
            file_list = []
            pc_dir = os.path.join(seq_dir, 'preprocess')
            # Get the list of files (full path) in the point cloud directory
            file_list = [os.path.join(pc_dir, f) for f in os.listdir(pc_dir) if f.endswith('.bin')]
            self.file_list.extend(file_list)
        # Setup the projection parameters
        self.projection = ScanProjection(proj_w=2048, proj_h=64)
        # Define the learning map for semantic labels
        # This map is used to convert the original labels to a smaller set of classes
        self.learning_map = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
            30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
            51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
            99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}
        # Create a mapping array with size large enough to cover the largest key
        self.max_key = max(self.learning_map.keys())
        self.map_array = np.zeros((self.max_key + 1,), dtype=np.int32)
        # Fill the mapping array with the learning map values
        for key, value in self.learning_map.items():
            self.map_array[key] = value
            
    # Read the point cloud data from binary files
    @staticmethod
    def readPCD(path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 9) # 9 channels: x, y, z, intensity, flag, R, G, B, label
        return pcd
  
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pc_path = self.file_list[idx]

        # Load binary data
        pc = self.readPCD(pc_path)  # x, y, z, intensity
        img, label = self.projection.doProjection(pc) # shape [H, W, C]
        # Map the labels using the learning map
        label = self.map_array[label]  # map to smaller set of classes
        img = torch.tensor(img).permute(2, 0, 1).float()  # to [C, H, W]
        label = torch.tensor(label).long()                # [H, W]
        # Normalize the tensor
        mean = torch.tensor([12.12, 10.88, 0.23, -1.04, 0.21])
        std = torch.tensor([12.32, 11.47, 6.91, 0.86, 0.16])
        img = (img - mean[:, None, None]) / std[:, None, None]
        return img, label
        