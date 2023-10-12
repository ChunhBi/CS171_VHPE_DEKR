# -*- coding: utf-8 -*-
import torch.utils.data
import os
import random
import pandas as pd
import numpy as np
import pickle as pkl



class HybridDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, seq_len):
        super(HybridDataset).__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len

        imu_ids = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 36, 37, 38, 39, 12, 10, 9]
        l = [[i * 21 + 5, i * 21 + 2, i * 21 + 3, i * 21 + 4, i * 21 + 6, i * 21 + 7, i * 21 + 8, i * 21 + 9,
              i * 21 + 10, i * 21 + 11] for i in [imu_ids[j] for j in [2, 5, 9, 13]]]
        self.valid_ids = sum(l, [])

        self.datamap = self.load_data(data_dir)
        self.cur_seq_idx = 0
        self.seq_list = self.generate_list()
        random.shuffle(self.seq_list)
        self.length = len(self.seq_list)

    def __iter__(self):
        return self

    def __next__(self):
        key, view, s, d = self.seq_list[self.cur_seq_idx]
        sample = self.datamap[key][0][s:d], self.datamap[key][1][view][s:d], self.datamap[key][2][s:d]
        self.cur_seq_idx += 1
        if self.cur_seq_idx >= self.length - 1:
            self.cur_seq_idx = 0
            random.shuffle(self.seq_list)
            raise StopIteration()
        return sample

    def load_data(self, root_dir):
        datamap = {}
        gen = os.walk(root_dir)

        for parent_dir, dir_list, file_list in gen:
            for file in file_list:
                if ".csv" in file:
                    csv_path = os.path.join(parent_dir, file)
                    data_path = os.path.join(parent_dir, str(int(file.split('.')[0][-3:])))
                    print(data_path)
                    gt_path = os.path.join(data_path, "gt.txt")
                    syn_path = os.path.join(data_path, "syn.txt")

                    syn = self.load_syn(syn_path)
                    first_rgb, gt = self.load_gt_seq(gt_path)
                    imu = pd.read_csv(csv_path, dtype=np.float32).values[syn[0] + (first_rgb - syn[1]) * 4::4,
                          self.valid_ids]
                    length = min(gt.shape[0], imu.shape[0])

                    vibes = self.load_vibe(os.path.join(parent_dir, file.split('.')[0][-1]), first_rgb, length)

                    gt = gt[:length, :]
                    imu = imu[:length, :]

                    datamap[csv_path] = [imu, vibes, gt]
        return datamap

    def load_gt_seq(self, path):
        data = []
        with open(path) as f:
            lines = f.readlines()
            data.append(int(lines[0].strip().split(' ')[0]))
            gt = []
            for line in lines:
                gt.append(np.array(line.strip().split(' ')[1:], dtype=np.float32)[None, :])
            gt = np.concatenate(gt, 0)
            data.append(gt)
        return data

    def load_syn(self, path):
        syn = []
        with open(path) as f:
            line = f.readline().strip().split(" ")
            syn.append(int(line[0]))
            syn.append(int(line[1]))
        return syn

    def load_vibe(self, parent_dir, first_rgb, length):
        vibes = []
        g = os.walk(parent_dir)
        for item in g:
            view_list = item[1]
            break

        for view in view_list:
            vibe_path = os.path.join(parent_dir, view, "vibe_output.pkl")
            vibe_dict = pkl.load(open(vibe_path, 'rb'))[1]
            vibes.append(vibe_dict['pose'][first_rgb:first_rgb + length])
        return vibes

    def generate_list(self):
        seq_list = []
        for key in self.datamap.keys():
            for view, vibe in enumerate(self.datamap[key][1]):
                start = 0
                end = start + self.seq_len
                while end < self.datamap[key][0].shape[0]:
                    seq_list.append((key, view, start, end))
                    start = end
                    end = start + self.seq_len
        return seq_list
