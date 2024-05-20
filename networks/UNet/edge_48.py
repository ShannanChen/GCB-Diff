import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm3d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm3d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, conv_z, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g_z = conv_z(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + torch.pow(g_z, 2))
    return torch.sigmoid(g) * input


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(y, 2, stride=2)

        return x, y

def get_sobel_3d(in_chan, out_chan):
    # Define the Sobel filters for x, y, and z directions in 3D
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)

    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_z = np.array([
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[-1, -2, -1],
         [-2, -4, -2],
         [-1, -2, -1]],
    ]).astype(np.float32)

    # Reshape the filters to include batch and channel dimensions
    filter_x = filter_x.reshape((1, 1, 1, 3, 3))
    filter_y = filter_y.reshape((1, 1, 1, 3, 3))
    filter_z = filter_z.reshape((1, 1, 3, 3, 3))

    # Repeat the filters for the number of input and output channels
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_z = np.repeat(filter_z, in_chan, axis=1)
    filter_z = np.repeat(filter_z, out_chan, axis=0)

    # Convert numpy arrays to torch tensors
    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_z = torch.from_numpy(filter_z)

    # Create torch parameters, indicating they should not be updated during training
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    filter_z = nn.Parameter(filter_z, requires_grad=False)

    # Define the 3D convolutional layers with the Sobel filters as weights
    conv_x = nn.Conv3d(in_chan, out_chan, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
    conv_y = nn.Conv3d(in_chan, out_chan, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
    conv_z = nn.Conv3d(in_chan, out_chan, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)

    # Assign the Sobel filters to the convolutional layers
    conv_x.weight = filter_x
    conv_y.weight = filter_y
    conv_z.weight = filter_z

    # Create sequential models for each direction including batch normalization
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm3d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm3d(out_chan))
    sobel_z = nn.Sequential(conv_z, nn.BatchNorm3d(out_chan))

    return sobel_x, sobel_y, sobel_z


class ESAM(nn.Module):
    def __init__(self, in_channels):
        super(ESAM, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm3d(in_channels)
        self.ban = nn.BatchNorm3d(1)
        self.sobel_x1, self.sobel_y1, self.sobel_z1 = get_sobel_3d(in_channels, 1)

    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, self.sobel_z1, x)
        y = F.relu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        y = F.relu(self.ban(y))

        return y


class Edgenet(nn.Module):
    def __init__(self):
        super(Edgenet, self).__init__()
        in_chan = 4
        self.down1 = Downsample_block(in_chan, 48)
        self.down2 = Downsample_block(48, 96)
        self.down3 = Downsample_block(96, 192)
        self.down4 = Downsample_block(192, 384)
        self.conv1 = nn.Conv3d(384, 768, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(768)
        self.up = ESAM(1)
        self.up1 = ESAM(96)
        self.up2 = ESAM(192)
        self.up3 = ESAM(384)
        self.up4 = ESAM(768)

    def forward(self, x, feature=False):
        x, y1 = self.down1(x)   # 48
        x, y2 = self.down2(x)   # 96
        x, y3 = self.down3(x)   # 192
        x, y4 = self.down4(x)   # 384
        y5 = F.relu(self.bn1(self.conv1(x)))    # 768

        out1 = self.up1(y2)
        out2 = self.up2(y3)
        out2 = F.interpolate(out2, scale_factor=2, mode='trilinear', align_corners=True)
        out3 = self.up3(y4)
        out3 = F.interpolate(out3, scale_factor=4, mode='trilinear', align_corners=True)
        out4 = self.up4(y5)
        out4 = F.interpolate(out4, scale_factor=8, mode='trilinear', align_corners=True)
        out = out1 + out2
        out = self.up(out)
        out = out + out3
        out = self.up(out)
        out = out + out4
        out = self.up(out)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=True)

        if feature:
            return y1, y2, y3, y4, y5, out
        else:
            return out
