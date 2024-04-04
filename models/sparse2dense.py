import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparse2Dense(nn.Module):
    def __init__(self):
        super(Sparse2Dense, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = conv_block(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        mid = self.middle(pool2)

        up2 = self.up2(mid)
        up2, enc2 = self.prepare_for_concat(up2, enc2)
        cat2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        up1, enc1 = self.prepare_for_concat(up1, enc1)
        cat1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(cat1)

        out = self.final_conv(dec1)
        padded_out = F.pad(out, (0, 0, 2, 0))
        return padded_out
    
    def prepare_for_concat(self, tensor1, tensor2):
        _, _, h1, w1 = tensor1.size()
        _, _, h2, w2 = tensor2.size()
        min_H = min(h1, h2)
        min_W = min(w1, w2)
        tensor1 = tensor1[:, :, :min_H, :min_W]
        tensor2 = tensor2[:, :, :min_H, :min_W]
        return tensor1, tensor2



