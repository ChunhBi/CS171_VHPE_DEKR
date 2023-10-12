# -*- coding: utf-8 -*-
import torch
import torch.utils.data
import math
import os
import shutil
import random
import pandas as pd
import numpy as np
import pickle as pkl
import json
from scipy.spatial.transform import Rotation as R

CSV_IMU_IDS = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 36, 37, 38, 39, 12, 10, 9]
IMU13_IDS = [CSV_IMU_IDS[i] for i in [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 15, 17]]

used_imu = [2, 5, 8, 10]

OP_21_JOINTS = [
    'OP LWrist', 'OP RWrist', 'OP LAnkle', 'OP RAnkle',
    'OP LElbow', 'OP RElbow', 'OP LKnee', 'OP RKnee',
    'OP LShoulder', 'OP RShoulder',
    'OP LHip', 'OP MidHip', 'OP RHip',
    'OP Nose', 'OP Neck',
    'OP LEye', 'OP REye', 'OP LEar', 'OP REar',
    'OP LBigToe', 'OP RBigToe',
]

MARKER21 = [
    'OP MidHip', 'OP Neck', 'OP LHip', 'OP RHip', 'Head', 'OP LShoulder', 'OP RShoulder', 'OP LKnee', 'OP RKnee',
    'OP Nose', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LElbow', 'OP RElbow', 'OP LAnkle', 'OP RAnkle',
    'OP LWrist', 'OP RWrist', 'OP LBigToe', 'OP RBigToe',
]

MARKER21_ID_MAP = {MARKER21[i]: i for i in range(len(MARKER21))}

MAP_MARKER21_TO_VIBE21 = [MARKER21_ID_MAP[key] for key in OP_21_JOINTS]


class HybridDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, seq_len):
        super(HybridDataset).__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len

        l = [[i * 21 + 2, i * 21 + 3, i * 21 + 4, i * 21 + 5, i * 21 + 9, i * 21 + 10, i * 21 + 11] for i in
             [IMU13_IDS[j] for j in used_imu]]
        self.valid_ids = sum(l, [])

        self.datamap = self.load_data(data_dir)
        self.cur_seq_idx = 0
        self.seq_list = self.generate_list()
        random.shuffle(self.seq_list)
        self.length = len(self.seq_list)
        self.shuffle = True


    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        key, view, s, d = self.seq_list[self.cur_seq_idx]

        first_cam_frame = self.datamap[key]['first_cam_frame']

        Ri2w = self.datamap[key]['Ri2w']
        Rs2b = self.datamap[key]['Rs2b']

        imu = self.datamap[key]['imu'][s:d]
        gt_transpose = self.datamap[key]['gt_transpose'][s:d]
        gt_shape = self.datamap[key]['gt_shape']

        Gw2c_list = self.datamap[key]['Gw2c_list']
        K_list = self.datamap[key]['K_list']
        view_name_list = self.datamap[key]['view_name_list']

        joints2d_list = [self.datamap[key]['joints2d_list'][i][s:d].transpose([0, 2, 1]) for i in
                         range(len(view_name_list))]

        proj_list = []
        for i in range(len(view_name_list)):
            K_ori = K_list[i]
            K = np.zeros((K_ori.shape[0], K_ori.shape[1] + 1), dtype=np.float32);
            K[:, :-1] = K_ori
            proj = K.dot(Gw2c_list[i].dot(np.linalg.inv(Gw2c_list[view])))
            proj_list.append(proj)

        Gw2c_allview = np.concatenate([item[None, ...] for item in Gw2c_list], 0)
        K_allview = np.concatenate([item[None, ...] for item in K_list], 0)
        joints2d_allview = np.concatenate([item[None, ...] for item in joints2d_list], 0)
        proj_allview = np.concatenate([item[None, ...] for item in proj_list], 0)

        sample = {
            'Ri2w': Ri2w,
            'Rs2b': Rs2b,
            'imu': imu,
            'gt_transpose': gt_transpose,
            'gt_shape': gt_shape,

            'Gw2c_allview': Gw2c_allview,
            # 'K_list': K_list,
            'view_name_list': view_name_list,
            'joints2d_allview': joints2d_allview,
            'proj_allview': proj_allview,
            # 'vibe': self.datamap[key]['vibe'][view][s:d],

            'view': view,
            'key': key,
            'interval': [s, d],
            'first_cam_frame': first_cam_frame,
        }
        self.cur_seq_idx += 1
        if self.cur_seq_idx >= self.length - 1:
            self.cur_seq_idx = 0
            if self.shuffle:
                random.shuffle(self.seq_list)
            raise StopIteration()
        return sample

    def load_data(self, root_dir):
        datamap = {}
        gen = os.walk(root_dir)

        for parent_dir, dir_list, file_list in gen:
            for file in file_list:
                if ".csv" in file and "chr" not in file:
                    csv_path = os.path.join(parent_dir, file)
                    data_path = os.path.join(parent_dir, str(int(file.split('.')[0][-3:])))

                    pose_path = os.path.join(data_path, "pose.txt")
                    shape_path = os.path.join(data_path, "shape.txt")
                    syn_path = os.path.join(data_path, "syn.txt")
                    s2b_path = os.path.join(data_path, "S2B.txt")
                    i2w_path = os.path.join(data_path, "I2C.txt")
                    #w2c_path = os.path.join(data_path, "calibration.json")
                    w2c_path = r"E:\Mocap\auto\calibration.json"

                    Ri2w = self.load_Ri2w(i2w_path)
                    Rs2b = self.load_Rs2b(s2b_path)
                    Gw2c_list, K_list, view_name_list = self.load_Gw2c(w2c_path)

                    syn = self.load_syn(syn_path)
                    first_cam_frame, gt_transpose = self.load_gt_seq(pose_path)

                    gt_shape = self.load_shape(shape_path)

                    imus = pd.read_csv(csv_path, dtype=np.float32).values
                
                    first_imu_frame = syn[0] + (first_cam_frame - syn[1]) * syn[2] / 59.94

                    imu_list = []
                    for idx in range(gt_transpose.shape[0]):
                        imu_list.append(imus[round(first_imu_frame + idx * syn[2] / 59.94):round(
                            first_imu_frame + idx * syn[2] / 59.94) + 1, self.valid_ids])
                    imus = np.concatenate(imu_list, 0)

                    print(gt_transpose.shape)

                    length = min(gt_transpose.shape[0], imus.shape[0])

                    # vibes = self.load_vibes(data_path, view_list, first_frame, length)
                    joints2d_list = self.load_joints2d(data_path, view_name_list, length)

                    gt_transpose = gt_transpose[:length, :]
                    imus = imus[:length, :]
                    datamap[csv_path] = {
                        'first_cam_frame': first_cam_frame,
                        'view_name_list': view_name_list,
                        'Ri2w': Ri2w,
                        'Rs2b': Rs2b,
                        'imu': imus,

                        'Gw2c_list': Gw2c_list,
                        'K_list': K_list,
                        # 'vibe': vibes,
                        'joints2d_list': joints2d_list,
                        'gt_transpose': gt_transpose,
                        'gt_shape': gt_shape
                    }

        return datamap

    def load_Ri2w(self, path):
        Ri2w = []
        with open(path) as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                line = line.strip().split()
                Ri2w.append(line)
        Ri2w = np.array(Ri2w, dtype=np.float32)
        return Ri2w

    def load_Rs2b(self, path):
        Rs2b_list = []
        with open(path) as f:
            lines = f.readlines()
            lines = lines[1:]
            Rs2b = []
            for i, line in enumerate(lines):
                line = line.strip().split()
                Rs2b.append(line)
                if (i + 1) % 3 == 0:
                    Rs2b = np.array(Rs2b, dtype=np.float32)
                    Rs2b_list.append(Rs2b)
                    Rs2b = []
        Rs2b_list = [Rs2b_list[i][None, :, :] for i in used_imu]
        Rs2b = np.concatenate(Rs2b_list, 0)
        return Rs2b

    def load_Gw2c(self, path):
        Gw2c_list = []
        K_list = []
        with open(path, 'r') as f:
            data = json.load(f)
        for key in data.keys():
            K = data[key]["K"]
            K = np.array(K, dtype=np.float32).reshape(3, 3)
            K_list.append(K)

            RT = data[key]["RT"]
            RT.extend([0, 0, 0, 1])
            RT = np.array(RT, dtype=np.float32).reshape(4, 4)
            Gw2c_list.append(RT)

        return Gw2c_list, K_list, list(data.keys())

    def load_gt_seq(self, path):
        with open(path) as f:
            lines = f.readlines()
            first_cam_frame = int(lines[0].strip().split(' ')[0])
            gt = []
            for line in lines:
                gt.append(np.array(line.strip().split(' ')[1:], dtype=np.float32)[None, :])
            gt = np.concatenate(gt, 0)
        return first_cam_frame, gt

    def load_shape(self, path):
        with open(path) as f:
            line = f.readline()
            shape = np.array(line.strip().split(' '), dtype=np.float32)

        return shape

    def load_syn(self, path):
        syn = []
        with open(path) as f:
            line = f.readline().strip().split(" ")
            syn.append(int(line[0]))
            syn.append(int(line[1]) - 1)
            if len(line) == 3:
                syn.append(int(line[2]))
            else:
                syn.append(120)
        return syn

    def load_vibes(self, parent_dir, view_name_list, first_rgb, length):
        vibes = []
        for view_folder in view_name_list:
            vibe_path = os.path.join(parent_dir, view_folder, "vibe_output.pkl")
            vibe_dict = pkl.load(open(vibe_path, 'rb'))[1]
            vibes.append(vibe_dict['pose'][first_rgb:first_rgb + length])
        return vibes

    def load_joints2d(self, parent_dir, view_name_list, length):
        joints2d_allview = []
        for view_folder in view_name_list:
            joints2d = []
            path = os.path.join(parent_dir, view_folder, "joints2d.txt")
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    joint2d = np.array(line.strip().split(" ")[1:], dtype=np.float32).reshape(3, -1)
                    joints2d.append(joint2d[None, :, :])
            joints2d = np.concatenate(joints2d, 0)
            joints2d_allview.append(joints2d[:length, :, MAP_MARKER21_TO_VIBE21])
        return joints2d_allview

    def generate_list(self):
        seq_list = []
        for key in self.datamap.keys():
            for view, view_name in enumerate(self.datamap[key]['view_name_list']):
                start = 0
                end = start + self.seq_len
                while end < self.datamap[key]['imu'].shape[0]:
                    seq_list.append((key, view, start, end))
                    start = end
                    end = start + self.seq_len
        return seq_list


class SimulationDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, seq_len):
        super(SimulationDataset).__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len

        self.datamap = self.load_data(data_dir)
        self.cur_seq_idx = 0
        self.seq_list = self.generate_list()
        random.shuffle(self.seq_list)
        self.length = len(self.seq_list)

    def __iter__(self):
        return self

    def __next__(self):
        key, s, d = self.seq_list[self.cur_seq_idx]
        sample = self.datamap[key][s:d]
        self.cur_seq_idx += 1
        if self.cur_seq_idx >= self.length - 1:
            self.cur_seq_idx = 0
            random.shuffle(self.seq_list)
            raise StopIteration()
        return sample

    def __len__(self):
        return self.length

    def load_data(self, root_dir):
        datamap = {}
        gen = os.walk(root_dir)

        for parent_dir, dir_list, file_list in gen:
            for file in file_list:
                if ".npy" in file:
                    framerate = int(file.split('.')[0].split('_')[-1])

                    motion_path = os.path.join(parent_dir, file)
                    motion = np.load(motion_path).astype(np.float32)

                    sample_frames = []
                    i = 0
                    while round(framerate * i / 60.) < motion.shape[0]:
                        sample_frames.append(int(round(framerate * i / 60.)))
                        i += 1

                    datamap[motion_path] = motion[sample_frames, :]
        return datamap

    def generate_list(self):
        seq_list = []
        for key in self.datamap.keys():
            start = 0
            end = start + self.seq_len
            while end < self.datamap[key].shape[0]:
                seq_list.append((key, start, end))
                start = end
                end = start + self.seq_len
        return seq_list