#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:28:51 2022

@author: Rodrigo
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from collections import namedtuple

class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet
    """
    def __init__(self,  num_filters = 64, inputLayer=False):
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

class ResNet(nn.Module):
    """
    ResNet
    
    Source: https://arxiv.org/abs/1512.03385
    """
    def __init__(self, num_filters=64):
        """
        Args:
          num_filters: Number of filter in the covolution
        """
        super(ResNet, self).__init__()
        
        # self.conv_first = nn.Conv2d(1, num_filters, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = ResidualBlock(num_filters, inputLayer=True)
        self.block2 = ResidualBlock(num_filters)
        self.block3 = ResidualBlock(num_filters)
        self.block4 = ResidualBlock(num_filters)
                
        self.conv_last = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        
        # out = self.conv_first(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        
        identity = x
        
        out = self.block1(x)
        
        out = self.block2(out)
                
        out = self.block3(out)
        
        out = self.block4(out)
                
        out = self.conv_last(out)
                
        out = self.relu(out + identity)
        
        return out

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
      batch = 2 * torch.sqrt(batch + 3./8. + self.sigma_e**2)

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
      batch = (batch/2)**2 + 1/4*torch.sqrt(torch.tensor(3/2))*batch**-1 - \
      11/8*batch**-2 + 5/8*torch.sqrt(torch.tensor(3/2))*batch**-3 - 1/8 - self.sigma_e**2 

      batch[batch<0] = 0

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

          
        omega1 = torch.sqrt((lambda_e * ((yhat-self.tau)/self.r_fct)+self.sgm**2)/ \
                (self.r_fct*lambda_e*((yhat-self.tau)/self.r_fct)+self.sgm**2))
          
        omega2 = (1./self.r_fct)-omega1

        # Weighted sum of reduced img and denoised img
        img_rest = omega1*(img_ld - self.tau) + omega2*(yhat-self.tau) + self.tau

        return img_rest
    
class ResNetModified(nn.Module):
    def __init__(self, tau, sigma_e, red_factor, maxGAT, minGAT, num_filters=64):
        super(ResNetModified, self).__init__()
        
        self.conv_first = nn.Conv2d(1, num_filters, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_filters, 1, 3, 1, 1)
        
        self.block1 = ResidualBlock(num_filters)
        self.block2 = ResidualBlock(num_filters)
        self.block3 = ResidualBlock(num_filters)
        self.block4 = ResidualBlock(num_filters)
        self.relu = nn.ReLU(inplace=True)

        self.iVST = iVST(sigma_e)
        self.VST = VST(tau, sigma_e)
        self.wSum = weightedSum(tau, sigma_e, red_factor)
        
        self.minGAT = minGAT
        self.maxGAT = maxGAT
        
    def forward(self, x):
        
        # ---- Strat building model
        
        data = x[:,0:1,:,:]
        lamb = x[:,1:2,:,:]

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

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = torch.cat([X, X, X], dim=1)
        #X = normalize_batch(X)
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class Vgg16_NoMaxPooling(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_NoMaxPooling, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice = torch.nn.Sequential()
        for x in range(23):
            if x not in [4, 9, 16]: # remove max pooling
                self.slice.add_module(str(x), vgg_pretrained_features[x])
    
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = torch.cat([X, X, X], dim=1)

        h = self.slice(X) # PL4 without max pooling

        return h

class RED_CNN(nn.Module):
    def __init__(self, tau, sigma_e, red_factor, maxGAT, minGAT, in_channels=1, out_channels=96, num_layers=5, kernel_size=5, padding=0):
        super(RED_CNN, self).__init__()

        encoder = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        decoder = [nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        for _ in range(num_layers):
            encoder.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            decoder.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
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

    def forward(self, x: torch.Tensor):

        # ---- Strat building model

        data = x[:, 0:1, :, :]
        lamb = x[:, 1:2, :, :]

        # GAT Layer
        out = self.VST(data, lamb)

        # Scale the GAT signal to [0-1]
        out = (out - self.minGAT) / self.maxGAT

        # RED
        residuals = []
        for block in self.encoder:
            residuals.append(out)
            out = F.relu(block(out), inplace=True)
        for residual, block in zip(residuals[::-1], self.decoder[::-1]):
            out = F.relu(block(out) + residual, inplace=True)
        # -----

        # Re-scale
        out = ((out * self.maxGAT) + self.minGAT)

        # Inverse GAT
        out = self.iVST(out)

        # Weighted Residual
        out = self.wSum(out, data, lamb)

        return out