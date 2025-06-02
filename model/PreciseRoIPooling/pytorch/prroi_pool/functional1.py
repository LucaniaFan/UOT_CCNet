import torch
import torch.nn.functional as F

def prroi_pool2d(input, rois, pooled_height, pooled_width, spatial_scale):
    """
    Pure Python implementation of Precise RoI Pooling
    """
    batch_size, num_channels, data_height, data_width = input.size()
    num_rois = rois.size(0)
    
    # Normalize coordinates
    rois = rois.clone()
    rois[:, 1:].mul_(spatial_scale)
    
    # Calculate grid points
    grid_h = torch.linspace(0, 1, pooled_height, device=input.device)
    grid_w = torch.linspace(0, 1, pooled_width, device=input.device)
    
    # Create output tensor
    output = torch.zeros(num_rois, num_channels, pooled_height, pooled_width, device=input.device)
    
    for i in range(num_rois):
        roi = rois[i]
        batch_idx = int(roi[0])
        x1, y1, x2, y2 = roi[1:5]
        
        # Calculate bin size
        bin_h = (y2 - y1) / pooled_height
        bin_w = (x2 - x1) / pooled_width
        
        # Calculate grid points for this ROI
        grid_y = y1 + grid_h * (y2 - y1)
        grid_x = x1 + grid_w * (x2 - x1)
        
        # Sample points
        for ph in range(pooled_height):
            for pw in range(pooled_width):
                # Get the four nearest points
                y = grid_y[ph]
                x = grid_x[pw]
                
                # Calculate weights
                y_low = int(y)
                y_high = min(y_low + 1, data_height - 1)
                x_low = int(x)
                x_high = min(x_low + 1, data_width - 1)
                
                # Bilinear interpolation
                ly = y - y_low
                lx = x - x_low
                hy = 1 - ly
                hx = 1 - lx
                
                # Get values
                v1 = input[batch_idx, :, y_low, x_low]
                v2 = input[batch_idx, :, y_low, x_high]
                v3 = input[batch_idx, :, y_high, x_low]
                v4 = input[batch_idx, :, y_high, x_high]
                
                # Interpolate
                w1 = hy * hx
                w2 = hy * lx
                w3 = ly * hx
                w4 = ly * lx
                
                output[i, :, ph, pw] = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    
    return output

def prroi_pool2d_grad(input, rois, grad_output, pooled_height, pooled_width, spatial_scale):
    """
    Pure Python implementation of Precise RoI Pooling gradient
    """
    batch_size, num_channels, data_height, data_width = input.size()
    num_rois = rois.size(0)
    
    # Normalize coordinates
    rois = rois.clone()
    rois[:, 1:].mul_(spatial_scale)
    
    # Initialize gradient
    grad_input = torch.zeros_like(input)
    
    # Calculate grid points
    grid_h = torch.linspace(0, 1, pooled_height, device=input.device)
    grid_w = torch.linspace(0, 1, pooled_width, device=input.device)
    
    for i in range(num_rois):
        roi = rois[i]
        batch_idx = int(roi[0])
        x1, y1, x2, y2 = roi[1:5]
        
        # Calculate bin size
        bin_h = (y2 - y1) / pooled_height
        bin_w = (x2 - x1) / pooled_width
        
        # Calculate grid points for this ROI
        grid_y = y1 + grid_h * (y2 - y1)
        grid_x = x1 + grid_w * (x2 - x1)
        
        # Sample points
        for ph in range(pooled_height):
            for pw in range(pooled_width):
                # Get the four nearest points
                y = grid_y[ph]
                x = grid_x[pw]
                
                # Calculate weights
                y_low = int(y)
                y_high = min(y_low + 1, data_height - 1)
                x_low = int(x)
                x_high = min(x_low + 1, data_width - 1)
                
                # Bilinear interpolation
                ly = y - y_low
                lx = x - x_low
                hy = 1 - ly
                hx = 1 - lx
                
                # Get gradient
                grad = grad_output[i, :, ph, pw]
                
                # Accumulate gradients
                grad_input[batch_idx, :, y_low, x_low] += grad * hy * hx
                grad_input[batch_idx, :, y_low, x_high] += grad * hy * lx
                grad_input[batch_idx, :, y_high, x_low] += grad * ly * hx
                grad_input[batch_idx, :, y_high, x_high] += grad * ly * lx
    
    return grad_input