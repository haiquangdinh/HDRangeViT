import torch
import numpy as np
import os
from torch.utils.data import Dataset

### DataLoader
class WaymoSegmentationDataset(Dataset):
    def __init__(self, root_dir, training):
        self.root_dir = root_dir
        self.file_list = []
        self.training = training  # Set to True for training dataset, False for validation
        file_list = []
        pc_dir = os.path.join(root_dir, 'preprocess')
        # Get the list of files (full path) in the point cloud directory
        file_list = [os.path.join(pc_dir, f) for f in os.listdir(pc_dir) if f.endswith('.bin')]
        self.file_list.extend(file_list)
            
    # Read the point cloud data from binary files
    @staticmethod
    def readPCD(path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(64, 2650, 10) # 10 channels: range, x, y, z, intensity, flag, R, G, B, label
        # crop it to 48, 480, 10
        pcd = pcd[8:56, 1085:1565, :]  # crop to 48, 480, 10
        return pcd
  
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pc_path = self.file_list[idx]

        # Load binary data
        pc = self.readPCD(pc_path)  # 10 channels: range, x, y, z, intensity, flag, R, G, B, label
        feats = pc[:, :, :9]  # Use the first 9 channels as features
        label = pc[:, :, 9]  # Use the last channel as label

        # Data augmentation: random translation and rotation
        if self.training:
            # Random translation (shift in H and W) (probability = 50%)
            if np.random.rand() < 0.5:
                max_trans = 10  # pixels
                trans_h = np.random.randint(-max_trans, max_trans + 1)
                trans_w = np.random.randint(-max_trans, max_trans + 1)
                feats = np.roll(feats, shift=trans_h, axis=0)
                feats = np.roll(feats, shift=trans_w, axis=1)
                label = np.roll(label, shift=trans_h, axis=0)
                label = np.roll(label, shift=trans_w, axis=1)
            # Random rotation (±5 degrees) (probability = 50%)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-5, 5)
                feats, label = self.rotate(feats, label, angle)
            # if np.random.rand() < 0.5:  # Random vertical flip
            #     feats = np.flip(feats, axis=0)
            #     label = np.flip(label, axis=0)
        feats = torch.tensor(feats).permute(2, 0, 1).float()  # to [C, H, W]
        label = torch.tensor(label).long()                # [H, W]

        return feats, label

    def rotate(self, feats, label, angle):
        # Rotate feats and label by given angle (in degrees)
        import cv2
        h, w = feats.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Rotate each channel
        feats_rot = np.stack([cv2.warpAffine(feats[:, :, c], rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for c in range(feats.shape[2])], axis=2)
        label_rot = cv2.warpAffine(label, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return feats_rot, label_rot