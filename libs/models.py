#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import torch
import torch.nn.functional as F
from torch import nn


class VST(nn.Module):
    """
    Generalized Anscombe VST
    """

    def __init__(self, tau, sigma_e):
        super(VST, self).__init__()

        self.tau = tau
        self.sigma_e = sigma_e

    def forward(self, batch, lambda_e):
        # Subtract offset and divide it by the gain of the quantum noise
        batch = (batch - self.tau) / lambda_e

        # Apply GAT (Generalized Anscombe VST)
        batch = 2 * torch.sqrt(batch + 3. / 8. + self.sigma_e ** 2)

        return batch


class iVST(nn.Module):
    """
    Inverse GAT - Closed-form approximation
    """

    def __init__(self, sigma_e):
        super(iVST, self).__init__()

        self.sigma_e = sigma_e

    def forward(self, batch):
        # Closed-form approximation of the exact unbiased inverse
        batch = (batch / 2) ** 2 + 1 / 4 * torch.sqrt(torch.tensor(3 / 2)) * batch ** -1 - \
                11 / 8 * batch ** -2 + 5 / 8 * torch.sqrt(torch.tensor(3 / 2)) * batch ** -3 - 1 / 8 - self.sigma_e ** 2

        batch[batch < 0] = 0

        return batch


class weightedSum(nn.Module):
    """

    """

    def __init__(self, tau, sigma_e, red_factor):
        super(weightedSum, self).__init__()

        self.tau = tau
        self.sgm = sigma_e
        self.r_fct = red_factor

    def forward(self, yhat, img_ld, lambda_e):
        # Multply it by the gain of the quantum noise and add the offset
        yhat = (yhat * lambda_e) + self.tau

        omega1 = torch.sqrt((lambda_e * ((yhat - self.tau) / self.r_fct) + self.sgm ** 2) / \
                            (self.r_fct * lambda_e * ((yhat - self.tau) / self.r_fct) + self.sgm ** 2))

        omega2 = (1. / self.r_fct) - omega1

        # Weighted sum of reduced img and denoised img
        img_rest = omega1 * (img_ld - self.tau) + omega2 * (yhat - self.tau) + self.tau

        return img_rest


'''  
    --------------
        Resnet
    --------------
'''


class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet
    """

    def __init__(self, num_filters=64, inputLayer=False):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(ResidualBlock, self).__init__()

        in_filters = num_filters
        if inputLayer:
            in_filters = 1

        self.conv1 = nn.Conv2d(in_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResResidualBlock(nn.Module):
    """
    Basic residual block for ResNet
    """

    def __init__(self, num_filters=64, inputLayer=False):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(ResResidualBlock, self).__init__()

        in_filters = num_filters
        if inputLayer:
            in_filters = 1

        self.conv1 = nn.Conv2d(in_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet
    
    Source: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, tau, sigma_e, red_factor, maxGAT, minGAT, num_filters=64):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(ResNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.block1 = ResidualBlock(num_filters, inputLayer=True)
        self.block2 = ResidualBlock(num_filters)
        self.block3 = ResidualBlock(num_filters)
        self.block4 = ResidualBlock(num_filters)

        self.conv_last = nn.Conv2d(num_filters + 1, 1, kernel_size=3, stride=1, padding=1)

        self.iVST = iVST(sigma_e)
        self.VST = VST(tau, sigma_e)
        self.wSum = weightedSum(tau, sigma_e, red_factor)

        self.minGAT = minGAT
        self.maxGAT = maxGAT

    def forward(self, x):
        # ---- Strat building model

        data = x[:, 0:1, :, :]
        lamb = x[:, 1:2, :, :]

        # GAT Layer
        out_vst = self.VST(data, lamb)

        # Scale the GAT signal to [0-1]
        out_vst = (out_vst - self.minGAT) / self.maxGAT

        # Resnet
        out = self.block1(out_vst)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.conv_last(torch.cat([out, out_vst], dim=1))
        out = self.relu(out)

        # Re-scale
        out = ((out * self.maxGAT) + self.minGAT)

        # Inverse GAT
        out = self.iVST(out)

        # Weighted Residual
        out = self.wSum(out, data, lamb)

        return out


class ResResNet(nn.Module):
    def __init__(self, tau, sigma_e, red_factor, maxGAT, minGAT, num_filters=64):
        super(ResResNet, self).__init__()

        self.conv_first = nn.Conv2d(1, num_filters, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_filters, 1, 3, 1, 1)

        self.block1 = ResResidualBlock(num_filters)
        self.block2 = ResResidualBlock(num_filters)
        self.block3 = ResResidualBlock(num_filters)
        self.block4 = ResResidualBlock(num_filters)
        self.relu = nn.ReLU(inplace=True)

        self.iVST = iVST(sigma_e)
        self.VST = VST(tau, sigma_e)
        self.wSum = weightedSum(tau, sigma_e, red_factor)

        self.minGAT = minGAT
        self.maxGAT = maxGAT

    def forward(self, x):
        # ---- Start building model

        data = x[:, 0:1, :, :]
        lamb = x[:, 1:2, :, :]

        # GAT Layer
        out_vst = self.VST(data, lamb)

        # Scale the GAT signal to [0-1]
        out_vst = (out_vst - self.minGAT) / self.maxGAT

        # First conv
        out1 = self.conv_first(out_vst)

        out2 = self.block1(out1)

        out3 = self.block2(out2)

        out3 = self.relu(out3 + out1)

        out4 = self.block3(out3)

        out5 = self.block4(out4)

        out5 = self.relu(out5 + out3)

        # Last conv
        out6 = self.conv_last(out5)

        # Relu        
        out6 = self.relu(out6 + out_vst)

        # Re-scale
        out6 = ((out6 * self.maxGAT) + self.minGAT)

        # Inverse GAT
        out6 = self.iVST(out6)

        # Weighted Residual
        out = self.wSum(out6, data, lamb)

        return out


class ResResNet_SDC(nn.Module):
    def __init__(self, num_filters=64):
        super(ResResNet_SDC, self).__init__()

        self.conv_first = nn.Conv2d(1, num_filters, 3, 1, 1)

        self.block1 = ResidualBlock(num_filters)
        self.block2 = ResidualBlock(num_filters)
        self.block3 = ResidualBlock(num_filters)
        self.block4 = ResidualBlock(num_filters)

        self.relu = nn.ReLU(inplace=True)

        self.conv_last = nn.Conv2d(num_filters, 1, 3, 1, 1)

    def forward(self, x):
        identity = x
        out1 = self.conv_first(x)

        out2 = self.block1(out1)

        out3 = self.block2(out2)

        out3 = self.relu(out3 + out1)

        out4 = self.block3(out3)

        out5 = self.block4(out4)

        out5 = self.relu(out5 + out3)

        out = self.conv_last(out5)

        out = self.relu(identity + out)

        return out


'''  
    --------------
        UNet
    --------------
'''


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, activation, mid_channels=None, use_bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_bn:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                activation
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation, use_bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, activation, bilinear=False, use_bn=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, activation, in_channels // 2, use_bn=use_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2(nn.Module):
    def __init__(self, tau, sigma_e, red_factor, maxGAT, minGAT, num_filters=64, n_channels=1, n_classes=1,
                 bilinear=False, residual=False, activation_type="relu", use_bn=True):
        super(UNet2, self).__init__()

        activation = nn.ReLU(inplace=True)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96 * 2, activation, use_bn=use_bn)
        self.down2 = Down(96 * 2, 96 * 4, activation, use_bn=use_bn)

        self.up1 = Up(96 * 4, 96 * 2, activation, use_bn=use_bn)
        self.up2 = Up(96 * 2, 96 * 1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

        self.iVST = iVST(sigma_e)
        self.VST = VST(tau, sigma_e)
        self.wSum = weightedSum(tau, sigma_e, red_factor)

        self.minGAT = minGAT
        self.maxGAT = maxGAT

    def forward(self, x):
        # ---- Strat building model

        data = x[:, 0:1, :, :]
        lamb = x[:, 1:2, :, :]

        # GAT Layer
        out_vst = self.VST(data, lamb)

        # Scale the GAT signal to [0-1]
        out_vst = (out_vst - self.minGAT) / self.maxGAT

        # UNET
        x1 = self.inc(out_vst)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        x = x + out_vst

        # Re-scale
        out6 = ((x * self.maxGAT) + self.minGAT)

        # Inverse GAT
        out6 = self.iVST(out6)

        # Weighted Residual
        out = self.wSum(out6, data, lamb)

        return out


class UNet2_SDC(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, residual=False, activation_type="relu", use_bn=True):
        super(UNet2_SDC, self).__init__()

        activation = nn.ReLU(inplace=True)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96 * 2, activation, use_bn=use_bn)
        self.down2 = Down(96 * 2, 96 * 4, activation, use_bn=use_bn)

        self.up1 = Up(96 * 4, 96 * 2, activation, use_bn=use_bn)
        self.up2 = Up(96 * 2, 96 * 1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        if self.residual:
            x = input + x
        return x


'''  
    --------------
        RED
    --------------
'''


class RED(nn.Module):
    def __init__(self, tau, sigma_e, red_factor, maxGAT, minGAT, num_filters=64, in_channels=1, out_channels=96,
                 num_layers=5, kernel_size=5, padding=0):
        super(RED, self).__init__()

        encoder = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        decoder = [
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        for _ in range(num_layers):
            encoder.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            )
            decoder.append(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            )

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.__init_weights()

        self.iVST = iVST(sigma_e)
        self.VST = VST(tau, sigma_e)
        self.wSum = weightedSum(tau, sigma_e, red_factor)

        self.minGAT = minGAT
        self.maxGAT = maxGAT

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, x):

        # ---- Strat building model

        data = x[:, 0:1, :, :]
        lamb = x[:, 1:2, :, :]

        # GAT Layer
        out_vst = self.VST(data, lamb)

        # Scale the GAT signal to [0-1]
        out_vst = (out_vst - self.minGAT) / self.maxGAT

        # RED
        residuals = []
        x = out_vst
        for block in self.encoder:
            residuals.append(x)
            x = F.relu(block(x), inplace=True)
        for residual, block in zip(residuals[::-1], self.decoder[::-1]):
            x = F.relu(block(x) + residual, inplace=True)

        # Relu 
        out6 = x
        # out6 = F.relu(x + out_vst)

        # Re-scale
        out6 = ((out6 * self.maxGAT) + self.minGAT)

        # Inverse GAT
        out6 = self.iVST(out6)

        # Weighted Residual
        out = self.wSum(out6, data, lamb)

        return out


class RED_SDC(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, num_layers=5, kernel_size=5, padding=0):
        super(RED_SDC, self).__init__()
        encoder = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        decoder = [
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        for _ in range(num_layers):
            encoder.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            )
            decoder.append(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            )
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        residuals = []
        for block in self.encoder:
            residuals.append(x)
            x = F.relu(block(x), inplace=True)
        for residual, block in zip(residuals[::-1], self.decoder[::-1]):
            x = F.relu(block(x) + residual, inplace=True)
        return x
