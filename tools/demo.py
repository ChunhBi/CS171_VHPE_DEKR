# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on DEKR.
# (https://github.com/HRNet/DEKR)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import csv
import os
import shutil
import time
import sys
sys.path.append("../lib")

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import offset_to_pose
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate
from dataset.transforms import FLIP_CONFIG

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

def get_pose_estimation_prediction(cfg, model, img_sq, transforms):
    # size at scale 1.0
    with torch.no_grad():

        scale=cfg.TEST.SCALE_FACTOR[0]
        img_sq_resized=[]
        base_sizes=[]
        resize_scares=[]
        for image in img_sq:
            base_size, center, _ = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0)
            base_sizes.append(base_size)
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )
            resize_scares.append(scale_resized)
            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()
            img_sq_resized.append(image_resized) 
            
        image_resized=torch.cat(img_sq_resized,dim=0).to(CTX)
        print(image_resized.shape)
        heatmap,offset=model(image_resized)
        #print(offset.shape)
        result=[]
        if cfg.TEST.FLIP_TEST:
            if 'coco' in cfg.DATASET.DATASET:
                flip_index_heat = FLIP_CONFIG['COCO_WITH_CENTER']
                flip_index_offset = FLIP_CONFIG['COCO']
            else:
                raise ValueError('Please implement flip_index \
                for new dataset: %s.' % cfg.DATASET.DATASET)

            image_resized_flip=torch.flip(image_resized, [3])
            image_resized_flip[:, :, :, :-3]=image_resized_flip[:, :, :, 3:]
            heatmap_flip, offset_flip = model(image_resized_flip)
            heatmap_flip = torch.flip(heatmap_flip, [3])
            heatmap = (heatmap + heatmap_flip[:, flip_index_heat, :, :])/2.0
        for j in range(len(img_sq)):
            poses = []
            heatmap_sum = 0
            posemap=offset_to_pose(offset[[j],:,:,:],flip=False)
            
            if cfg.TEST.FLIP_TEST:
                posemap_flip = offset_to_pose(offset_flip[[j],:,:,:], flip_index=flip_index_offset)
                posemap = (posemap + torch.flip(posemap_flip, [3]))/2.0

            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap[[j],:,:,:], posemap, scale
            ) 
        
            heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
            for i in range(len(poses)):
                if type(poses[i])==np.ndarray:
                    poses[i]=torch.from_numpy(poses[i]).to(CTX)
                
            poses, scores = pose_nms(cfg, heatmap_sum, poses)
            print(scores)
            if len(scores) == 0:
                return []
            else:
                if cfg.TEST.MATCH_HMP:
                    poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

                final_poses = get_final_preds(
                    poses, center, resize_scares[j], base_sizes[j]
                )

            final_results = []
            final_score=[]
            for i in range(len(scores)):
                final_results.append(final_poses[i])
                final_score.append(scores[i])
                    
            result.append((final_results,final_score))
        

    return result

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=60)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir

def draw_skeleton(img,pred):
    for i, coords in enumerate(pred):
        for j,coord in enumerate(coords):
            x_coord, y_coord = int(coord[0]), int(coord[1])
            cv2.circle(img, (x_coord, y_coord), 4, (255, 255, 0), 2)

        coords_=[(int(coord[0]),int(coord[1])) for coord in coords]
        #print(coords_)
        centre=(int((coords_[5][0]+coords[6][0])/2),int((coords_[5][1]+coords[6][1])/2))
        
        bones=[[coords_[0],coords_[1],(0,1)],[coords_[0],coords_[2],(0,2)],[coords_[1],coords_[3],(1,3)],
        [coords_[2],coords_[4],(2,4)],[coords_[0],centre,(0,0)],[centre,coords_[5],(5,5)],[coords_[6],centre,(6,6)],
        [coords_[5],coords_[7],(5,7)],[coords_[6],coords_[8],(6,8)],[coords_[7],coords_[9],(7,9)],
        [coords_[8],coords_[10],(8,10)],[coords_[11],centre,(11,11)],[coords_[12],centre,(12,12)],[coords_[11],coords_[13],(11,13)],
        [coords_[12],coords_[14],(12,14)],[coords_[13],coords_[15],(13,15)],[coords_[14],coords_[16],(14,16)]]
        for bone in bones:
            cv2.line(img,bone[0],bone[1],(255, 255, 0),thickness=6)
            
    return img

def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []
    txt_output_rows=[]

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (frame_width, frame_height))

    count = 0
    precision = 0
    sqe_len=10
    sqe_img=[]
    sqe_img_bgr=[]
    all_score = []
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        if not ret:
            break
        count += 1
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sqe_img.append(image_rgb)
        sqe_img_bgr.append(image_bgr)
        if count%sqe_len!=0:
            continue
        print(len(sqe_img))
        results=get_pose_estimation_prediction(cfg,pose_model,sqe_img,transforms=pose_transform)

        for i,result in enumerate(results):
            pred,score=result
            img=draw_skeleton(sqe_img_bgr[i],pred)
            img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
            cv2.imwrite(img_file, img)
            outcap.write(img)
            all_score.append(score[0])
            if score[0] > 0.5 :
                precision += 1

        sqe_img=[]
        sqe_img_bgr=[]
    with open(os.path.join(args.outputDir, "output.txt"), "w") as f:
        for score in all_score:
            f.write(str(score) + "\n")
    final_score = np.mean(np.array(all_score))
    print(final_score)
    print(precision/count)
    outcap.release()

if __name__ == '__main__':
    main()