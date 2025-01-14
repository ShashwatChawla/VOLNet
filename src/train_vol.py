import sys
sys.path.append('/ocean/projects/cis220039p/schawla1/VOLNet/src')

import torch

from configs.base_config import BaseConfig
from dataloader.dataloader import get_dataloader, process_data
from model.vol_net import VOLNet
from training.trainer import Trainer
from training.losses import compute_loss

def main():
    config = BaseConfig()

    train_dataloader = get_dataloader(config, 'train')
    val_dataloader = get_dataloader(config, 'val')

    model = VOLNet(config)

    if config.load_pretrained_volnet:
        volnet_checkpoint = config.volnet_checkpoint
        volnet_checkpoint = torch.load(volnet_checkpoint)
        volnet_weights = volnet_checkpoint['model'] if 'model' in volnet_checkpoint else volnet_checkpoint
        model.load_state_dict(volnet_weights, strict=True)
        print("Loaded pretrained volnet weights")

    if config.load_pretrained_flow:
        flow_checkpoint = config.flow_checkpoint
        flow_checkpoint = torch.load(flow_checkpoint)
        flow_weights = flow_checkpoint['model'] if 'model' in flow_checkpoint else flow_checkpoint
        model.flow_net.load_state_dict(flow_weights, strict=True)
        print("Loaded pretrained flow weights")
    
    if config.freeze_flow_net:
        model.flow_net.eval()
        for param in model.flow_net.parameters():
            param.requires_grad = False
        print("Froze flow net")
    
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        loss_fn=compute_loss,
        data_process_fn=process_data
    )

    trainer.train()
    

if __name__ == '__main__':
    main()
