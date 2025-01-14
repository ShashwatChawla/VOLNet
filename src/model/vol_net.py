import sys
sys.path.append('/ocean/projects/cis220039p/schawla1/VOLNet/src')

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flow.gmflow.gmflow.gmflow import GMFlow
from model.output_heads import RotationHead, TranslationHead
from model.vit import VisionTransformer
from model.resnet import ResNet
from configs.base_config import BaseConfig
from dataloader.dataloader import get_dataloader, process_data


class VOLNet(nn.Module):
    def __init__(self, config):
        super(VOLNet, self).__init__()
        self.config = config

        if not self.config.use_gt_flow:
            self.flow_net = GMFlow(
                num_scales=1, 
                upsample_factor=8, 
                feature_channels=128,
                attention_type='swin', 
                num_transformer_layers=6, 
                ffn_dim_expansion=4, 
                num_head=1
            )

        self.pose_net = None
        if config.pose_net_type == 'vit': 
            self.pose_net = VisionTransformer(
                img_size=640,
                patch_size=16,
                in_channels=14,
                embed_dim=768,
                num_heads=12,
                num_layers=12,
                mlp_dim=3072,
                out_dim=1024,
                dropout=0.1
            )
        elif config.pose_net_type == 'resnet':
            self.pose_net = ResNet(in_dim=14)
        else:
            raise ValueError("Invalid pose_net_type. Must be one of ['vit', 'resnet']")

        self.rotation_head = RotationHead(1024, 2048, 4)
        self.translation_head = TranslationHead(1024, 2048, 3)


    def forward(self, x, return_flow=False):
        img1 = x["images"][:, 0]    # retrieve the first image
        img2 = x["images"][:, 1]    # retrieve the second image

        lidar1 = x["pointmaps"][:, 0]     # retrieve the first lidar pointmap
        lidar2 = x["pointmaps"][:, 1]     # retrieve the second lidar pointmap

        if self.config.use_gt_flow:
            flow = x["flows"].squeeze(1)    # retrieve the ground truth flow

        else:
            # Flow model
            flow_output = self.flow_net(img1, img2, attn_splits_list = [2],
                                                corr_radius_list = [-1],
                                                prop_radius_list = [-1], 
                                                pred_bidir_flow  =  False)
            flow = flow_output['flow_preds'][0]     # retrieve the forward flow

        # Append flow + lidar
        fused_data = torch.cat([flow, lidar1, lidar2], dim=1)

        # Append original image
        fused_data = torch.cat([fused_data, img1, img2], dim=1)

        # Vision Transformer
        decoder_output = self.pose_net(fused_data)

        # Output heads
        rotation = self.rotation_head(decoder_output)
        translation = self.translation_head(decoder_output)

        output = {"rotation": rotation, "translation": translation}

        if return_flow:
            output["flow"] = flow
        
        return output


# Example usage
if __name__ == "__main__":

    # Create a config object.
    config = BaseConfig()

    net = VOLNet(config)

    # Create a dataloader object.
    dataloader = get_dataloader(config)

    data = dataloader.load_sample()
    data = process_data(data)

    rotation, translation = net(data)
    print(f"Rotation output shape: {rotation.shape}")  # Should output (B, 4)
    print(f"Translation output shape: {translation.shape}")  # Should output (B, 3)