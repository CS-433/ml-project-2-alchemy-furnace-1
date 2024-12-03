from torchvision.models import vit_b_16
from torch import nn

class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head

    def forward(self, x):
        x = self.vit(x)
        return x

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)  # Reshape to (B, C, 1, 1)
        x = self.decoder(x)
        return x
class VITdecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VITdecoder, self).__init__()
        self.decoder =vit_b_16(pretrained=True)
        self.decoder.heads = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1)
                

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)  # Reshape to (B, C, 1, 1)
        x = self.decoder(x)
        return x
# class Patchify(nn.Module):
#     def __init__(self, patch_size=16):
#         super(Patchify, self).__init__()
#         self.patch_size = patch_size

#     def forward(self, x):
#         B, C, H, W = x.shape
#         p = self.patch_size
#         assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"
#         x = x.reshape(B, C, H // p, p, W // p, p)
#         x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)
#         return x
    
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vit_b_16
import collections
class Patchify(nn.Module):
    def __init__(self, patch_size=16):
        super(Patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        patch_size = self.patch_size
        assert height % patch_size == 0 and width % patch_size == 0, "Image dimensions must be divisible by patch size"
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, patch_size * patch_size * channels)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerBlocks(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_dim, dropout=0.1):
        super(TransformerBlocks, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)
        return src


import torch
import torch.nn as nn
import torchvision.models as models

class ViTEncoderDecoder(nn.Module):
    def __init__(self, image_size=400, patch_size=16, in_channels=3, embed_dim=512, num_heads=8, num_layers=5, mlp_dim=2048):
        super(ViTEncoderDecoder, self).__init__()
        self.encoder1 = self.CNNBlock(in_channels, 64, namePrefix="enc1")
        self.encoder2 = self.CNNBlock(64, 128, namePrefix="enc2")
        self.encoder3 = self.CNNBlock(128, 256, namePrefix="enc3")
        self.encoder4 = self.CNNBlock(256, 512, namePrefix="enc4")
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Use ResNet as feature extractor
        # resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers (avgpool and fc)
        # resnet_out_channels = 512  # For ResNet-18, the output channels from last convolutional block is 512
        
        # Ensure output matches the embedding dimension
        # self.conv_proj = nn.Conv2d(resnet_out_channels, embed_dim, kernel_size=1)
        self.conv_proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        # Calculate the number of patches after feature extraction
        num_patches = (image_size // 32) ** 2

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2, embed_dim))
        self.bottleneck = self.CNNBlock(512,512, namePrefix="b")
        # Encoder and decoder
        # self.encoder = TransformerBlocks(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim)
        # self.decoder = TransformerBlocks(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim)
        self.transformer = TransformerBlocks(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim)
        # Upsampling layers
        self.upconv4 = nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2)
        self.decoder4 = self.CNNBlock(1024, 512, namePrefix="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.CNNBlock(512, 256, namePrefix="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.CNNBlock(256, 128, namePrefix="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.CNNBlock(128, 64, namePrefix="dec1")
        
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)
        
        self.patch_size = patch_size
    @staticmethod
    def CNNBlock(num_channels_in, features, namePrefix):
        block_kernel_size = 3
        block_padding = 1
        conv1 = (namePrefix + "conv1",
                 nn.Conv2d(
                     in_channels=num_channels_in,
                     out_channels=features,
                     kernel_size=block_kernel_size,
                     padding=block_padding,
                     bias=False,
                 ))
        norm1 = (namePrefix + "norm1", nn.BatchNorm2d(num_features=features))
        relu1 = (namePrefix + "relu1", nn.ReLU(inplace=True))
        conv2 = (namePrefix + "conv2",
                 nn.Conv2d(
                     in_channels=features,
                     out_channels=features,
                     kernel_size=block_kernel_size,
                     padding=block_padding,
                     bias=False,
                 ))
        norm2 = (namePrefix + "norm2", nn.BatchNorm2d(num_features=features))
        relu2 = (namePrefix + "relu2", nn.ReLU(inplace=True))
        return nn.Sequential(collections.OrderedDict([conv1, norm1, relu1, conv2, norm2, relu2]))
    
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        p1 = self.maxpool(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.maxpool(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.maxpool(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.maxpool(e4)
        
        # Bottleneck: ViT
        # b = self.conv_proj(p4)  # Project to embedding dimension
        # batch_size, embed_dim, H, W = b.shape
        # b = b.flatten(2).permute(0, 2, 1)  # Flatten and prepare for Transformer (B, N, D)
        # # b = b + self.pos_embedding # Add positional embedding
        # # b = self.transformer(b)  # Transformer Encoder

        # b = b.permute(0, 2, 1).view(batch_size, embed_dim, H, W)  # Reshape back to 2D feature map
        b = self.bottleneck(p4)
        # Decoder path
        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat((d4, e4), dim=1))
        
        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((d3, e3), dim=1))
        
        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((d2, e2), dim=1))
        
        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((d1, e1), dim=1))
        
        return torch.sigmoid(self.conv_final(d1))  # Output (batch_size, 1, H, W)

