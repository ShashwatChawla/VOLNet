import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class NuScenesDataset(Dataset):
    def __init__(self, nusc, sample_indices):
        self.nusc = nusc
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample = self.nusc.sample[self.sample_indices[idx]]
        
        # Get left camera image
        camera_token = sample['data']['CAM_FRONT_LEFT']
        image_path = self.nusc.get_sample_data_path(camera_token)
        image = cv2.imread(image_path)
        
        # Get Velodyne LiDAR data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path = self.nusc.get_sample_data_path(lidar_token)
        pc = LidarPointCloud.from_file(lidar_path)
        
        # Get car pose
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose_token = lidar_data['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        
        return {
            'image': image,
            'lidar_points': pc.points,  # LiDAR points (Nx4) for Velodyne
            'pose': ego_pose,
        }


# Function to pad LiDAR data to the maximum size in the batch
def pad_lidar_data(lidar_points_list, max_len=None):
    if max_len is None:
        max_len = max([pc.shape[1] for pc in lidar_points_list])  # Find the maximum number of points (second dimension)
    
    # Pad each lidar point cloud to max_len (assuming each point has 4 attributes)
    padded_lidar = []
    for pc in lidar_points_list:
            # Convert numpy array to torch tensor
            pc_tensor = torch.tensor(pc).float()
            
            # Calculate padding size
            pad_size = max_len - pc_tensor.shape[1]
            
            # Pad along the second dimension (number of points), the pad value is 0
            if pad_size > 0:
                padded_pc = F.pad(pc_tensor, (0, pad_size), value=0)  # Padding only on the second dimension
            else:
                padded_pc = pc_tensor
            
            padded_lidar.append(padded_pc)
    
    # Stack into a single tensor
    return torch.stack(padded_lidar)


# Custom collate function for batching with padding
def collate_fn(batch):
    images = [item['image'] for item in batch]
    lidar_points = [item['lidar_points'] for item in batch]
    car_poses = [item['pose'] for item in batch]
    
    # Resize images (optional step, depending on model input size)
    images = [torch.tensor(cv2.resize(img, (224, 224))).permute(2, 0, 1).float() for img in images]
    
    # Pad lidar points to max length in the batch
    lidar_points = pad_lidar_data(lidar_points)
    
    # Convert car poses to tensors (you may need to extract translation/rotation as needed)
    car_poses = [torch.tensor([ego_pose['translation'] + ego_pose['rotation']]) for ego_pose in car_poses]
    
    return {
        'image': torch.stack(images),
        'lidar_points': lidar_points,
        'pose': torch.stack(car_poses)
    }


## Usage

# Create the dataset and dataloader
nusc = NuScenes(version='v1.0-mini', dataroot='/ocean/projects/cis220039p/shared/nuscenes', verbose=True)
sample_indices = [0, 1, 2, 3, 4]  # Just an example, use your actual sample indices
dataset = NuScenesDataset(nusc, sample_indices)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

# Test Batch
for batch in dataloader:
    print(batch['image'].shape)  # Batch of images (e.g., [batch_size, C, H, W])
    print(batch['lidar_points'].shape)  # Padded batch of lidar points
    print(batch['pose'])  
