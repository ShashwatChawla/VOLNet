from gmflow.gmflow.gmflow import GMFlow
import torch
import cv2

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift


def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0, add_arrows = False, new_shape = False): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    flownp is of shape (H, W, 2)

    The output is a numpy array of shape (H, W, 3) with values in range [0, 255], type uint8.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)


    # Scale to a larger shape before adding arrows, and then scale back.
    # bgr = cv2.resize(bgr, (bgr.shape[1] * 2, bgr.shape[0] * 2))
    flow_h, flow_w, _ = flownp.shape

    if new_shape:
        new_h = new_shape[0]
        new_w = new_shape[1]
    else:
        new_h = flownp.shape[0]
        new_w = flownp.shape[1]
    
    bgr = cv2.resize(bgr, (new_w, new_h))
    flownp = cv2.resize(flownp, (new_w , new_h))

    # Adjust the flow values.
    flownp[:, :, 0] = flownp[:, :, 0] * 2
    flownp[:, :, 1] = flownp[:, :, 1] * 2

    if add_arrows:
        for i in range(0, flownp.shape[0], new_h // 10):
            for j in range(0, flownp.shape[1], new_w // 10):
                cv2.arrowedLine(bgr, (j, i), (j + int(flownp[i, j, 0]), i + int(flownp[i, j, 1])), (255, 0, 0), 1)
        
    return bgr

model = GMFlow(num_scales=1, upsample_factor=8, feature_channels=128, attention_type='swin', num_transformer_layers=6, ffn_dim_expansion=4, num_head=1)

flow_net = GMFlow(
                    num_scales = 1,
                    upsample_factor = 8,
                    feature_channels = 128,
                    attention_type = 'swin',
                    num_transformer_layers = 6,
                    ffn_dim_expansion = 4,
                    num_head = 1
                    )

flow_net = flow_net.cuda()

flow_checkpoint_gfp = "/ocean/projects/cis220039p/pkachana/projects/tartanvo-fisheye-old/src/tartanvo_fisheye/networks/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth" # GMFlow checkpoint

checkpoint = torch.load(flow_checkpoint_gfp)
weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
flow_net.load_state_dict(weights, strict=True)

import matplotlib.pyplot as plt
import os 
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))
out_dir = os.path.join(curr_dir, 'output')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Load images
img1_path = '/ocean/projects/cis220039p/pkachana/projects/data/coalmine/Data_easy/P000/image_lcam_front/000000_lcam_front.png'
img2_path = '/ocean/projects/cis220039p/pkachana/projects/data/coalmine/Data_easy/P000/image_lcam_front/000001_lcam_front.png'
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)


# Convert images to tensors
img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

img1_tensor = img1_tensor.cuda()
img2_tensor = img2_tensor.cuda()

# Run images through the flow model
flow_output = flow_net(img1_tensor, img2_tensor, attn_splits_list = [2] ,
                                                    corr_radius_list = [-1],
                                                    prop_radius_list = [-1], 
                                                    pred_bidir_flow  =  False)

# Visualize the output
flow = flow_output['flow_preds'][0][0]

flow_np = flow.detach().cpu().numpy().transpose(1,2,0)
flow_vis = visflow(flow_np, add_arrows=True)

cv2.imwrite(os.path.join(out_dir, 'flow_vis.png'), flow_vis)





