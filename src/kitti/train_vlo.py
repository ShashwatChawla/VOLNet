# -*- coding:UTF-8 -*-

import os
import sys
import torch
import torch.utils.data
import numpy as np
import time
from tqdm import tqdm
from kitti_pytorch import OdometryDataset

def main():
    train_dir_list = [0, 1, 2, 3, 4, 5, 6]
    #train_dir_list = [4]
    test_dir_list = [7,8,9,10]
    eval_before_train = True
    eval_every_n_epoch = 100

    optimizer = 'Adam'
    learning_rate = 0.001
    max_epoch = 10000
    batch_size = 128 # as much as possible, depands on GPU mem
    
    # train set
    train_dataset = OdometryDataset()
    print(train_dataset[0])

    # train set
    train_dataset = None
    train_loader = None

    model = None
    loss_fn = None

    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=0.9)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0.0001)
        
    # optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=13,
    #                                             gamma=0.7, last_epoch=-1)

    train_losses = []
    epochs = []

    
    if eval_before_train == True:
        with torch.no_grad():
            score = eval_pose(model, test_dir_list)
        print('EPOCH {} scores: {:04f}'.format(epoch, score))

    for epoch in range(max_epoch):
        total_loss = 0
        total_data_number = 0
        optimizer.zero_grad()

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            torch.cuda.synchronize()
            start_train_one_batch = time.time()

            ## example data dimensions
            # rgb (B, H, W, 3)
            # lidar (B, n_points_lidar, 3)
            # gt_translation (B, 3) 
            # gt_rotation (B, 4) (assume quaternion)

            rgb, lidar, gt_translation, gt_rotation = data

            torch.cuda.synchronize()
            print('load_data_time: ', time.time() - start_train_one_batch)

            rgb = rgb.cuda().to(torch.float32)
            lidar = lidar.cuda().to(torch.float32)
            gt_translation = gt_translation.cuda().to(torch.float32)
            gt_rotation = gt_rotation.cuda().to(torch.float32)

            model = model.train()

            torch.cuda.synchronize()
            print('load_data_time + model_trans_time: ', time.time() - start_train_one_batch)

            pred_translation, pred_rotation = model(rgb, lidar)

            torch.cuda.synchronize()
            print('load_data_time + model_trans_time + forward ', time.time() - start_train_one_batch)


            loss = loss_fn(pred_translation, gt_translation, pred_rotation, gt_rotation)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            print('load_data_time + model_trans_time + forward + back_ward ', time.time() - start_train_one_batch)

            total_loss += loss.cpu().data * batch_size
            total_data_number += batch_size

        train_losses.append(total_loss/total_data_number)
        epochs.append(epoch)
        print('EPOCH {} train mean loss: {:04f}'.format(epoch, float(total_loss)))

        if (epoch+1) % eval_every_n_epoch == 0:
            with torch.no_grad():
                score = eval_pose(model, test_dir_list)
            print('EPOCH {} scores: {:04f}'.format(epoch, score))


if __name__ == '__main__':
    main()
