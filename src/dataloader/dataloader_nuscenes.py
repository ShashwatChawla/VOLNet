import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import einops
import os
import cv2
import numpy as np

# NuScenes Camera List:
# 0: CAM_FRONT
# 1: CAM_FRONT_LEFT
# 2: CAM_FRONT_RIGHT
# 3: CAM_BACK_LEFT
# 4: CAM_BACK_RIGHT
# 5: CAM_BACK
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4, 5]
MAX_LIDAR_PTS = 35000

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, cam_id, mode='mini', seq=0):
        super(NuScenesDataset, self).__init__()

        if mode == 'mini':
            dataroot = dataroot + 'mini/'
        elif mode == 'train':
            dataroot = dataroot + 'train/'
        elif mode == 'eval':
            dataroot = dataroot + 'eval/'
        

        # TODO: Adapt for multiple sequences
        self.data_path = os.path.join(dataroot, f"{seq:03d}/")
        self.timesteps = len(os.listdir(self.data_path + 'lidar_pose'))
        self.cam_id = cam_id

        self.pairs = self.create_pairs()
        
    def create_pairs(self):
        """
        Required to load pair of imgs, lidar_pts, poses, extrinsics...
        """
        pairs = []
        
        for t in range(self.timesteps):
            if t == (self.timesteps -1):
                break
            pair = [t, t + 1]
            pairs.append(pair)
        # print(f"Total pairs created: {len(pairs)}")    
        return pairs

    def get_cam_intrinsics(self):
        # fx, fy, cx, cy, distortions (0 in the case of nuScenes)
        return np.loadtxt(os.path.join(self.data_path, "intrinsics/", f"{self.cam_id}.txt"))
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pairs = self.pairs[idx] # load pairs
        
        images = self.load_images(pairs) 
        lidar_pts = self.load_lidar_pts(pairs)
        # ego_pose in camera frame
        cam_pose = self.load_cam_pose(pairs) 
        # ego_pose in lidar_frame
        lidar_pose = self.load_lidar_pose(pairs)
    
        return {
            'images': images, 
            'cam_pose': cam_pose,
            'lidar_pose': lidar_pose,
            'lidar_pts': lidar_pts
        }
    
    def load_images(self, pairs):
        images = []
        for idx in pairs:
            img = cv2.imread(os.path.join(self.data_path, "images/", f"{idx:03d}_{self.cam_id}.jpg")) 
            images.append(img)
        return np.stack(images, axis=0)
    
    def load_lidar_pts(self, pairs):
        
        lidar_pts_all = []
        for idx in pairs:
            lidar_filepath = os.path.join(self.data_path, "lidar/", f"{idx:03d}.bin") 
            lidar_pts = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 4)[:, :3]
            num_pts = lidar_pts.shape[0]

            # TODO: Replace by padding using a collate_func
            # Pad with zeros < than MAX_LIDAR_PTS
            if num_pts < MAX_LIDAR_PTS:
                pad_size = MAX_LIDAR_PTS - num_pts
                padding = np.zeros((pad_size, 3), dtype=np.float32)
                lidar_pts = np.vstack([lidar_pts, padding])  # Pad at the end

            # Truncate if the pts > MAX_LIDAR_PTS (just for safety: always less)
            elif num_pts > MAX_LIDAR_PTS:
                print(f"!!!!! Lidar pts greater that max pts :{num_pts} !!!!!!")
                lidar_pts = lidar_pts[:MAX_LIDAR_PTS, :]  
                
            lidar_pts_all.append(lidar_pts)
        
        return np.stack(lidar_pts_all, axis=0)

    def load_cam_pose(self, pairs):
        cam_poses = []
        for idx in pairs:
            pose = np.loadtxt(os.path.join(self.data_path, "extrinsics/", f"{idx:03d}_{self.cam_id}.txt"))
            cam_poses.append(pose)

        return np.stack(cam_poses, axis=0)

    def load_lidar_pose(self, pairs):
        lidar_poses = []
        for idx in pairs:
            pose = np.loadtxt(os.path.join(self.data_path, "lidar_pose/", f"{idx:03d}.txt")) # Only lidar-top
            lidar_poses.append(pose)

        return np.stack(lidar_poses, axis=0)



# Usage 
dataroot = '/ocean/projects/cis220039p/shared/nuscenes_full/processed/'

cam_id = 0
dataset = NuScenesDataset(dataroot, cam_id)
cam_intrinsics = dataset.get_cam_intrinsics()

dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

for batch in dataloader:
        print("### Aquired Data ###")
        print(f"Img Shape :{batch['images'].shape}")
        print(f"Lidar Shape :{batch['lidar_pts'].shape}")
        print(f"Cam_pose Shape :{batch['cam_pose'].shape}")
        print(f"Lidar_pose Shape :{batch['lidar_pose'].shape}")

exit()


        
