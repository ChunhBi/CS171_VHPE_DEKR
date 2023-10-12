import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PoolingNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, channel, stage_number):
        super(PoolingNet, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv1d(in_features, channel, kernel_size=1, stride=1, bias=True)
        self.expand_bn = nn.BatchNorm1d(channel, momentum=0.1)
        self.stage_number = stage_number
        self.conv_depth = len(kernel_size_set)
        layers = []

        for stage_index in range(0, stage_number):
            for conv_index in range(len(kernel_size_set)):
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(channel, channel, kernel_size_set[conv_index], stride_set[conv_index], dilation=1, bias=True),
                        nn.BatchNorm1d(channel, momentum=0.1)
                    )
                )

        self.shrink = nn.Conv1d(channel, out_features, kernel_size=1, stride=1, bias=True)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for stage_index in range(0, self.stage_number):
            output = 0
            for conv_index in range(self.conv_depth):
                output += F.adaptive_avg_pool1d(self.drop(self.relu(self.layers[stage_index*self.conv_depth + conv_index](x))), x.shape[-1])
            x = output
        x = self.shrink(x)
        data = {'fullpose_6d':torch.transpose(x, 1, 2)}
        return data