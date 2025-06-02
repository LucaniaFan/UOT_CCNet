import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class HeadDescriptorExtractor(nn.Module):
    def __init__(self, in_channels, feature_dim=256):
        super(HeadDescriptorExtractor, self).__init__()
        
        # Head localization network
        self.head_loc = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(in_channels, feature_dim//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, feature_dim//2, kernel_size=5, padding=2)
        
        # Spatial attention
        self.attention = SpatialAttention()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final head feature extraction
        self.head_feat = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Head localization
        head_map = self.head_loc(x)
        
        # Multi-scale feature extraction
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        
        # Concatenate multi-scale features
        multi_scale_feat = torch.cat([feat1, feat2], dim=1)
        
        # Apply spatial attention
        attention_weights = self.attention(multi_scale_feat)
        attended_feat = multi_scale_feat * attention_weights
        
        # Feature fusion
        fused_feat = self.fusion(attended_feat)
        
        # Extract final head features
        head_features = self.head_feat(fused_feat)
        
        return head_map, head_features, attention_weights