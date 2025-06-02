#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import torch
import torch.nn.functional as F
import torch.autograd as ag

__all__ = ['prroi_pool2d']


def prroi_pool2d(features, rois, pooled_height, pooled_width, spatial_scale):
    """
    Pure Python implementation of Precise RoI Pooling
    Args:
        features: Input features (N, C, H, W)
        rois: Region of interests (K, 5) [batch_idx, x1, y1, x2, y2]
        pooled_height: Height of pooled output
        pooled_width: Width of pooled output
        spatial_scale: Scale factor for ROI coordinates
    Returns:
        Pooled features (K, C, pooled_height, pooled_width)
    """
    # Ensure inputs are on the same device
    device = features.device
    rois = rois.to(device)
    
    # Get batch size and number of channels
    N, C, H, W = features.size()
    K = rois.size(0)
    
    # Normalize coordinates
    rois = rois.float()
    rois[:, 1:] = rois[:, 1:] * spatial_scale
    
    # Initialize output tensor
    output = torch.zeros((K, C, pooled_height, pooled_width), device=device)
    
    # Process each ROI
    for k in range(K):
        batch_idx = int(rois[k, 0])
        x1, y1, x2, y2 = rois[k, 1:]
        
        # Clamp coordinates
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        
        # Calculate grid points
        grid_h = torch.linspace(y1, y2, pooled_height, device=device)
        grid_w = torch.linspace(x1, x2, pooled_width, device=device)
        
        # Create grid for sampling
        grid_y, grid_x = torch.meshgrid(grid_h, grid_w)
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)
        
        # Normalize grid coordinates to [-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0
        
        # Sample features using grid_sample
        features_roi = features[batch_idx:batch_idx+1]
        output[k] = F.grid_sample(
            features_roi,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0)
    
    return output

