import torch.utils
import torch.utils.checkpoint
from .unet_mini_parts import *

class UNetMini(nn.Module):
    def __init__(self, n_channels):
        super(UNetMini, self).__init__()
        self.n_channels = n_channels

        self.inc = (DoubleConv_1(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.jump = (Jump(128, 256))
        self.down2 = (Down(256, 512))
        self.up = (Up(512, 256))
        self.outc = (OutConv(256, 1))


    def forward(self, x):
        # print(f"x:{x.shape}")
        x1 = self.inc(x)
        # print(f"x1:{x1.shape}")
        x2 = self.down1(x1)
        # print(f"x2:{x2.shape}")
        x3 = self.jump(x2)
        # print(f"x3:{x3.shape}")
        x4 = self.down2(x3)
        # print(f"x4:{x4.shape}")
        x = self.up(x4, x3)
        # print(f"x5:{x.shape}")
        logits = self.outc(x)
        # print(f"x6:{x.shape}")
        return logits
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.jump = torch.utils.checkpoint(self.jump)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.up = torch.utils.checkpoint(self.up)
        self.outc = torch.utils.checkpoint(self.outc)

class UNetMiniPro(nn.Module):
    def __init__(self, n_channels):
        super(UNetMiniPro, self).__init__()
        self.n_channels = n_channels

        self.inc = (DoubleConv_2(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.down5 = (Down(1024, 2048))
        self.up = (Up(2048, 1024))
        self.outc = (OutConv(1024, 1))


    def forward(self, x):

        x1 = self.inc(x)
        # print(f"x1:{x1.shape}")
        x2 = self.down1(x1)
        # print(f"x2:{x2.shape}")
        x3 = self.down2(x2)
        # print(f"x3:{x3.shape}")
        x4 = self.down3(x3)
        # print(f"x4:{x4.shape}")
        x5 = self.down4(x4)
        # print(f"x5:{x5.shape}")
        x6 = self.down5(x5)
        # print(f"x6:{x6.shape}")
        x = self.up(x6, x5)
        # print(f"x7:{x.shape}")
        logits = self.outc(x)
        # print(f"x8:{logits.shape}")
        return logits
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.up = torch.utils.checkpoint(self.up)
        self.outc = torch.utils.checkpoint(self.outc)