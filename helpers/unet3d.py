"""
A lightweight 3D U-Net for binary whole-tumour segmentation.

Encoder: at each level two 3x3x3 convolutions (InstanceNorm + LeakyReLU); the first convolution of every
level below the top has stride 2, which halves D, H and W. Decoder: a 2x2x2 transposed convolution doubles
the size again, the encoder features of the same level are concatenated (skip connection), and two more
convolutions follow. A 1x1x1 convolution maps the top-level features to one logit per voxel.

The network is fully convolutional, so it is trained on patches and applied to a whole volume with a
sliding window (helpers/inference.py). InstanceNorm is used instead of BatchNorm because the batch holds
only 2 patches, too few for stable batch statistics.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    """Two (conv 3x3x3 -> InstanceNorm -> LeakyReLU) layers; the first conv may downsample with stride 2."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class UNet3D(nn.Module):
    """3D U-Net: in_channels MRI channels -> out_channels logits, same spatial size as the input.

    widths  feature channels per level, top to bottom; len(widths) - 1 downsamplings, so every spatial
            size of the input must be divisible by 2 ** (len(widths) - 1).
    """

    def __init__(self, in_channels=4, out_channels=1, widths=(16, 32, 64, 128, 256)):
        super().__init__()
        self.widths = tuple(widths)
        self.encoder = nn.ModuleList()
        prev = in_channels
        for level, w in enumerate(widths):
            self.encoder.append(ConvBlock(prev, w, stride=1 if level == 0 else 2))
            prev = w
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for w_skip, w_below in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.upsample.append(nn.ConvTranspose3d(w_below, w_skip, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(2 * w_skip, w_skip))
        self.head = nn.Conv3d(widths[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = skips.pop()                          # bottleneck features
        for up, block in zip(self.upsample, self.decoder):
            x = up(x)
            x = block(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
