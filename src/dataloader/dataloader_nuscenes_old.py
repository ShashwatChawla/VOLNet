import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import einops
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import einops
import numpy as np

NED2CAM = np.array(
    [[0, 1, 0, 0], 
    [0, 0, 1, 0], 
    [1, 0, 0, 0], 
    [0, 0, 0, 1]], dtype=np.float32
)

def quat_to_rot(q_poses):
    translations = q_poses[:, :3]  # Extract translations
    quaternions = q_poses[:, 3:]  # Extract quaternions

    # Convert all quaternions to rotation matrices
    rotations = R.from_quat(quaternions).as_matrix()

    # Create transformation matrices
    n = translations.shape[0]
    pose_matrices = np.zeros((n, 4, 4))
    pose_matrices[:, :3, :3] = rotations
    pose_matrices[:, :3, 3] = translations
    pose_matrices[:, 3, 3] = 1

    return pose_matrices

def ned_to_cam(poses):
    # Similiarity transformation
    poses = NED2CAM @ poses @ np.linalg.inv(NED2CAM) 
    return poses



class NuScenesDataset(Dataset):
    def __init__(self, nusc, sample_indices):
        self.nusc = nusc
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        # TODO: Sample 2 points together
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
    # Find the maximum number of points (second dimension)
    if max_len is None:
        max_len = max([pc.shape[1] for pc in lidar_points_list])  
    
    # Pad each lidar point cloud to max_len (assuming each point has 4 attributes)
    padded_lidar = []
    for pc in lidar_points_list:
            
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
    poses = [item['pose'] for item in batch]
    
    # Resize images (optional step, depending on model input size)
    images = [torch.tensor(cv2.resize(img, (640, 640))).permute(2, 0, 1).float() for img in images]
    
    # Pad lidar points to max length in the batch
    lidar_points = pad_lidar_data(lidar_points)
    
    poses = [torch.tensor([ego_pose['translation'] + ego_pose['rotation']]) for ego_pose in poses]
    
    return {
        'image': torch.stack(images),
        'lidar_points': lidar_points,
        'pose': torch.stack(poses)
    }


def process_pose(poses) -> torch.Tensor:
    """
    Process the pose data.
    """

    # Rearrange the poses to move sequences to batch dimension.
    B, N, _ = poses.shape
    poses = einops.rearrange(poses, 'b n i -> (b n) i')

    # Convert the poses quaternions to rotation matrices.
    poses = quat_to_rot(poses)

    # Convert the poses from NED to camera coordinates.
    poses = ned_to_cam(poses)

    # Rearrange the poses back to the original shape.
    poses = einops.rearrange(poses, '(b n) i j -> b n i j', b=B, n=N)

    # Why??
    # Convert poses to be in the frame of the first camera.
    first_pose = poses[:, 0]
    first_pose_inv = np.linalg.inv(first_pose)
    first_pose_inv = np.repeat(first_pose_inv[:, np.newaxis], poses.shape[1], axis=1)
    poses = first_pose_inv @ poses

    # Convert the poses to torch tensors.
    poses = torch.tensor(poses, dtype=torch.float32)

    return poses


## Usage

# Create the dataset and dataloader
nusc = NuScenes(version='v1.0-mini', dataroot='/ocean/projects/cis220039p/shared/nuscenes', verbose=True)


all_sample_tokens = []
for scene in nusc.scene:
    first_sample_token = scene['first_sample_token']
    sample_token = first_sample_token
    while sample_token != '':
        all_sample_tokens.append(sample_token)
        sample_token = nusc.get('sample', sample_token)['next']

print(f"Total samples: {len(all_sample_tokens)}")

# not-too bad
exit()

# TODO: Alternative func to load data (inidices is not scalable)
sample_indices = [0, 1, 2, 3, 4]  # Example sample indices
dataset = NuScenesDataset(nusc, sample_indices)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

# Test Batch
for batch in dataloader:
    print("### Printing data ###")
    # Batch of images (e.g., [batch_size, C, H, W])
    print(batch['image'].shape)          
    print(batch['lidar_points'].shape)  
    print(batch['pose'])            
    batch['pose'] = process_pose(batch['pose'])
    print("Transformed pose")
    print(batch['pose'])            


