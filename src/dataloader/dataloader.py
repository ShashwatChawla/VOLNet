import os
import sys
sys.path.append('/ocean/projects/cis220039p/schawla1/VOLNet/src')

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import einops
import tartanair as ta

from configs.base_config import BaseConfig


INTRINSICS = torch.tensor(
    [[320., 0., 320.],
    [0., 320., 240.],
    [0., 0., 1.]]
)

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


def rot_to_quat(rot_poses):
    translations = rot_poses[:, :3, 3]  # Extract translations
    rotations = rot_poses[:, :3, :3]  # Extract rotations

    # Convert all rotation matrices to quaternions
    quaternions = R.from_matrix(rotations).as_quat()

    # Create transformation matrices
    n = translations.shape[0]
    pose_quats = np.zeros((n, 7))
    pose_quats[:, :3] = translations
    pose_quats[:, 3:] = quaternions

    return pose_quats
    

def ned_to_cam(poses):
    poses = NED2CAM @ poses @ np.linalg.inv(NED2CAM)
    return poses
   

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

    # Convert poses to be in the frame of the first camera.
    first_pose = poses[:, 0]
    first_pose_inv = np.linalg.inv(first_pose)
    first_pose_inv = np.repeat(first_pose_inv[:, np.newaxis], poses.shape[1], axis=1)
    poses = first_pose_inv @ poses

    # Convert the poses to torch tensors.
    poses = torch.tensor(poses, dtype=torch.float32)

    return poses


def unproject_depth_to_pointmap(batch):
    B, N, H, W = batch['depths'].shape

    depths = einops.rearrange(batch['depths'], 'b n h w -> (b n) h w')
    poses = einops.rearrange(batch['poses'], 'b n i j -> (b n) i j')

    # Create the pointmap.
    pointmaps = torch.zeros((B*N, H, W, 3), dtype=torch.float32)

    # Create the pixel grid.
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y = y.unsqueeze(0).expand(B*N, -1, -1).float()
    x = x.unsqueeze(0).expand(B*N, -1, -1).float()

    # Create the pixel coordinates.
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    pixel_coords = pixel_coords.reshape(-1, 3)

    # Create the camera parameters.
    cx, cy, fx, fy = INTRINSICS[0, 2], INTRINSICS[1, 2], INTRINSICS[0, 0], INTRINSICS[1, 1]
    cx = cx.unsqueeze(0).unsqueeze(1).expand(B*N, -1, -1)
    cy = cy.unsqueeze(0).unsqueeze(1).expand(B*N, -1, -1)
    fx = fx.unsqueeze(0).unsqueeze(1).expand(B*N, -1, -1)
    fy = fy.unsqueeze(0).unsqueeze(1).expand(B*N, -1, -1)

    # Compute 3D points.
    Z = depths
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    
    # Create the camera coordinates.
    pointmaps = torch.stack([X, Y, Z], dim=-1)

    # Transform to homogeneous coordinates.
    pointmaps = torch.cat([pointmaps, torch.ones_like(pointmaps[..., :1])], dim=-1)
    pointmaps = einops.rearrange(pointmaps, '(b n) h w c -> (b n) (h w) c', b=B, n=N) 

    # Transform to world coordinates.
    pointmaps = (poses @ pointmaps.transpose(1, 2)).transpose(1, 2)
    pointmaps = pointmaps[..., :3]

    # Reshape back to the original shape.
    pointmaps = einops.rearrange(pointmaps, '(b n) (h w) c -> b n h w c', b=B, n=N, h=H, w=W)

    return pointmaps


def get_dataloader(config, mode='train'):
    # Initialize TartanAir.
    ta.init(config.tartanair_data_root)

    if mode == 'train':
        # Create a train dataloader object.
        dataloader = ta.dataloader(env = config.train_envs,
                    difficulty = config.train_difficulties,
                    trajectory_id = config.train_trajectory_ids,
                    modality = config.modalities,
                    camera_name = config.camnames,
                    # new_image_shape_hw = config.new_image_shape_hw,
                    seq_length = config.seq_length,
                    subset_framenum = config.subset_framenum,
                    seq_stride = config.seq_stride,
                    frame_skip = config.frame_skip,
                    batch_size = config.batch_size,
                    num_workers = config.num_workers,
                    shuffle = config.shuffle,
                    verbose = True)

    elif mode == 'val':
        # Create a val dataloader object.
        dataloader = ta.dataloader(env = config.val_envs,
                    difficulty = config.val_difficulties,
                    trajectory_id = config.val_trajectory_ids,
                    modality = config.modalities,
                    camera_name = config.camnames,
                    # new_image_shape_hw = config.new_image_shape_hw,
                    seq_length = config.seq_length,
                    subset_framenum = config.subset_framenum,
                    seq_stride = config.seq_stride,
                    frame_skip = config.frame_skip,
                    batch_size = config.batch_size,
                    num_workers = config.num_workers,
                    shuffle = config.shuffle,
                    verbose = True)


    print("Dataloader created.")

    return dataloader


def resize_data(data, new_shape):
    new_H, new_W = new_shape
    B, N, C, H, W = data['images'].shape

    images = data['images']
    images = einops.rearrange(images, 'b n c h w -> (b n) c h w')
    images = torch.nn.functional.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
    data['images'] = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=N)

    pointmaps = data['pointmaps']
    pointmaps = einops.rearrange(pointmaps, 'b n c h w -> (b n) c h w')
    pointmaps = torch.nn.functional.interpolate(pointmaps, size=(new_H, new_W), mode='nearest')
    data['pointmaps'] = einops.rearrange(pointmaps, '(b n) c h w -> b n c h w', b=B, n=N)

    return data


def random_mask_data(
    batch: dict,
    keys: list[str],
    min_mask: float = 0.25, 
    max_mask: float = 0.75
) -> torch.Tensor:
    """
    Mask the data with random values between min_mask and max_mask.
    """
    B, N, C, H, W = batch[keys[0]].shape

    mask_percentage = torch.empty(B, N).uniform_(min_mask, max_mask)
    mask = torch.rand(B, N, 1, H, W) < mask_percentage.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    mask = mask.to(batch[keys[0]].device)

    for key in keys:
        batch[key] = batch[key].masked_fill(mask, 0)

    return batch


def random_pad_data(
    batch: dict,
    keys: list[str],
    min_pad: int = 0, 
    max_pad: int = 0.75, 
) -> torch.Tensor:
    """
    Mask out the edges of the data with random padding.
    """
    B, N, C, H, W = batch[keys[0]].shape

    max_pad = max_pad * 0.5 # max_pad is the percentage of the image to pad

    #TODO: use different padding along batches and sequences
    random_x_pad = torch.randint(0, int(W * max_pad), (1,))
    random_y_pad = torch.randint(0, int(H * max_pad), (1,))

    for key in keys:
        data = batch[key]

        data[:, :, :, :random_y_pad, :] = 0
        data[:, :, :, -random_y_pad:, :] = 0
        data[:, :, :, :, :random_x_pad] = 0
        data[:, :, :, :, -random_x_pad:] = 0
        
    return batch


def augment_data(batch):

    # Mask the data
    batch = random_mask_data(batch, ['pointmaps', 'masks'])
    batch = random_pad_data(batch, ['images', 'pointmaps', 'flows', 'masks'])

    return batch


def process_data(batch, new_size=None, device='cpu'):
    # Process the pose data.
    batch['poses'] = process_pose(batch['pose_lcam_front'])
    batch.pop('pose_lcam_front')

    # Save the motion quaternion between frames
    motions = torch.tensor(rot_to_quat(batch['poses'][:, 1]))  # the first frame should be the identity so second frame pose is the motion
    batch['rotations'] = motions[:, 3:].float()         # used to compute loss so convert to float
    batch['translations'] = motions[:, :3].float()      # used to compute loss so convert to float

    # Rename the image, depth, and flow data.
    batch['images'] = batch.pop('rgb_lcam_front')
    batch['images'] = batch['images'][..., [2,1,0]]  # convert images from BGR to RGB
    batch['images'] = batch['images'].permute(0, 1, 4, 2, 3)  # convert images to PyTorch format
    batch['images'] = batch['images'] / 255.0  # normalize images to [0, 1]

    batch['depths'] = batch.pop('depth_lcam_front')

    batch['flows'] = batch.pop('flow_lcam_front')
    batch['flows'] = batch['flows'].permute(0, 1, 4, 2, 3)  # convert flows to PyTorch format

    # Project the depth data to a pointmap.
    batch['pointmaps'] = unproject_depth_to_pointmap(batch)
    batch['pointmaps'] = batch['pointmaps'].permute(0, 1, 4, 2, 3)  # convert pointmaps to PyTorch format
    batch.pop('depths')

    if new_size is not None:
        # Resize the data.
        batch = resize_data(batch, new_size)

    batch['masks'] = torch.ones_like(batch['images'][:, :, 0:1])

    # Convert the data to the specified device.
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    batch = augment_data(batch)

    return batch


def main():
    # Create a config object.
    config = BaseConfig()

    # Create a dataloader object.
    dataloader = get_dataloader(config)

    # Iterate over the batches.
    for i in range(100):
        # Get the next batch.
        batch = dataloader.load_sample()
        batch = process_data(batch)

        for i in range(batch['images'].shape[0]):
            image = batch['images'][i].numpy()
            pointmap = batch['pointmaps'][i].numpy()

            cv2.imwrite('image_{i}.png', cv2.cvtColor(image*255.0, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(f'pointmap_{i}.npy', (int)((pointmap[i][:,:,2]/pointmap[i][:,:,2].max())*255.0))

        # np.savez(f'batch_{i}.npz', **batch)

    dataloader.stop_cachers()


if __name__ == '__main__':
    main()