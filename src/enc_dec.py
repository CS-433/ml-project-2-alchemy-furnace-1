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


class ViTEncoderDecoder(nn.Module):
    def __init__(self, image_size=400, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, num_layers=6, mlp_dim=2048):
        super(ViTEncoderDecoder, self).__init__()
        self.patchify = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # 替代线性投影
        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2, embed_dim))
        self.encoder = TransformerBlocks(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim)
        self.decoder = TransformerBlocks(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim)
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)  # 上采样操作
        self.output_layer = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.patch_size = patch_size
    def forward(self, x):
        # Patchify input image
        x = self.patchify(x)  # (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).permute(0, 2, 1)  # (batch_size, num_patches, embed_dim)
        x = x + self.pos_embedding  # Add position embedding
        x = x.permute(1, 0, 2)  # (num_patches, batch_size, embed_dim)
        
        # Encoder
        memory = self.encoder(x)  # Transformer Encoder
        
        # Decoder
        x = self.decoder(memory)
        
        # Reshape back to 2D
        x = x.permute(1, 2, 0).contiguous().view(x.size(1), -1, int(400 / 16), int(400 / 16))
        x = self.upsample(x)  # 上采样
        x = self.output_layer(x)
        # x = x.view(x.size(0), -1)
        return x


