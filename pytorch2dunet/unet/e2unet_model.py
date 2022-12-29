""" Full assembly of the parts to form the complete network """

from .e2unet_parts import *
import torch



class e2UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(e2UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (e2DoubleConv(n_channels, 64))
        self.down1 = (e2Down(64, 128))
        self.down2 = (e2Down(128, 256))
        self.down3 = (e2Down(256, 512))
        factor = 2 
        self.down4 = (e2Down(512, 1024 // factor))
        self.up1 = (e2Up(1024, 512 // factor))
        self.up2 = (e2Up(512, 256 // factor))
        self.up3 = (e2Up(256, 128 // factor))
        self.up4 = (e2Up(128, 64))
        self.outc = (e2OutConv(64, n_classes))
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)