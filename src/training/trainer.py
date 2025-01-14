import time
import os
import json
import sys
sys.path.append('/ocean/projects/cis220039p/schawla1/VOLNet/src')

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb

from configs.base_config import BaseConfig
from dataloader.dataloader import get_dataloader
from model.vol_net import VOLNet
from visualization.vis import visflow


class Trainer():

    def __init__(self, 
        config, 
        model, 
        train_loader,
        val_loader,
        loss_fn,
        data_process_fn = None,
        optimizer = None,
        scheduler = None,
        logger = None
    ):

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.data_process_fn = data_process_fn

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.optimizer = optimizer

        if scheduler is None:
            scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        self.scheduler = scheduler

        if logger is None:
            logger = wandb.init(
                project=config.project_name,
                # config=config.__dict__
            )
        self.logger = logger

        self.num_steps = self.config.num_steps
        self.val_freq = self.config.val_freq
        self.val_steps = self.config.val_steps

        self.log_freq = self.config.log_freq
        self.vis_freq = self.config.vis_freq
        self.val_vis_freq = self.config.val_vis_freq
        
        current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.ckpt_save_dir = f"{self.config.ckpt_save_dir}/{current_time}"
        os.makedirs(self.ckpt_save_dir, exist_ok=True)

        config_dict = config.__dict__
        config_dict['datetime'] = current_time
        with open(os.path.join(self.ckpt_save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
        self.model.to(self.device)

    
    def train_step(self, data, return_flow=False):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(data, return_flow=return_flow)
        loss = self.loss_fn(
            output['rotation'], 
            data['rotations'], 
            output['translation'], 
            data['translations'],
            alpha=self.config.loss_alpha,
            supervise_flow=self.config.supervise_flow,
            predicted_flow=output['flow'] if return_flow else None,
            gt_flow=data['flows'] if return_flow else None,
            mask=data['masks'] if return_flow else None
        )

        total_loss = loss['total_loss']
        total_loss.backward()
        self.optimizer.step()

        return loss, output
    
    @torch.no_grad()
    def val_step(self, data, return_flow=False):
        self.model.eval()

        output = self.model(data, return_flow=return_flow)
        loss = self.loss_fn(
            output['rotation'], 
            data['rotations'], 
            output['translation'], 
            data['translations'],
            alpha=self.config.loss_alpha,
            supervise_flow=self.config.supervise_flow,
            predicted_flow=output['flow'] if return_flow else None,
            gt_flow=data['flows'] if return_flow else None,
            mask=data['masks'] if return_flow else None
        )

        return loss, output
    

    def val(self, step):
        val_rot_loss = 0
        val_trans_loss = 0
        val_total_loss = 0

        for val_step in range(self.val_steps):
            val_batch = self.val_loader.load_sample()
            val_processed_batch = self.data_process_fn(val_batch, device=self.device)

            return_flow = (val_step % self.val_vis_freq == 0 or self.config.supervise_flow)

            val_loss, output = self.val_step(val_processed_batch, return_flow=return_flow)
            val_rot_loss += val_loss['rotation_loss']
            val_trans_loss += val_loss['translation_loss']
            val_total_loss += val_loss['total_loss']

            if val_step % self.val_vis_freq == 0:
                img1 = val_processed_batch['images'][0][0].cpu().numpy().transpose(1, 2, 0)
                img2 = val_processed_batch['images'][0][1].cpu().numpy().transpose(1, 2, 0)
                flow = output['flow'][0].cpu().numpy().transpose(1, 2, 0)
                flow_img = visflow(flow)
                gt_flow = val_processed_batch['flows'][0][0].cpu().numpy().transpose(1, 2, 0)
                gt_flow_img = visflow(gt_flow)

                self.logger.log({
                    "val_images": [
                        wandb.Image(img1, caption="Image 1"),
                        wandb.Image(img2, caption="Image 2"),
                        wandb.Image(flow_img, caption="Pred Flow"),
                        wandb.Image(gt_flow_img, caption="GT Flow")
                    ]
                })

        val_rot_loss /= self.val_steps
        val_trans_loss /= self.val_steps
        val_total_loss /= self.val_steps

        self.logger.log({
            "val_rotation_loss": val_rot_loss,
            "val_translation_loss": val_trans_loss,
            "val_total_loss": val_total_loss
        })

        torch.save(self.model.state_dict(), f"{self.ckpt_save_dir}/model_step_{step}.pt")


    def train(self):
        for step in tqdm.tqdm(range(self.num_steps)):
            batch = self.train_loader.load_sample()
            processed_batch = self.data_process_fn(batch, device=self.device)

            return_flow = (step % self.vis_freq == 0 or self.config.supervise_flow)
            loss, output = self.train_step(processed_batch, return_flow=return_flow)

            # self.scheduler.step()

            if step % self.log_freq == 0:
                log = {
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }
                for key, value in loss.items():
                    log[key] = value
                
                if step % self.vis_freq == 0:
                    img1 = processed_batch['images'][0][0].detach().cpu().numpy().transpose(1, 2, 0)
                    img2 = processed_batch['images'][0][1].detach().cpu().numpy().transpose(1, 2, 0)
                    flow = output['flow'][0].detach().cpu().numpy().transpose(1, 2, 0)
                    flow_img = visflow(flow)
                    gt_flow = processed_batch['flows'][0][0].detach().cpu().numpy().transpose(1, 2, 0)
                    gt_flow_img = visflow(gt_flow)

                    log["train_images"] = [
                            wandb.Image(img1, caption="Image 1"),
                            wandb.Image(img2, caption="Image 2"),
                            wandb.Image(flow_img, caption="Pred Flow"),
                            wandb.Image(gt_flow_img, caption="GT Flow")
                        ]

                self.logger.log(log)
            
            if step % self.val_freq == 0:
                self.val(step)    


if __name__ == '__main__':
    pass
