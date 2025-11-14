import torch
import torch.nn as nn
import MinkowskiEngine as ME


# Fallback for NVIDIA MinkowskiEngine fork (no ME.MinkowskiSequential exported)
if hasattr(ME, "MinkowskiSequential"):
    MinkowskiSequential = ME.MinkowskiSequential
else:
    class MinkowskiSequential(nn.Module):
        def __init__(self, *modules):
            super().__init__()
            # allow list/tuple or *args
            if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
                modules = modules[0]
            self.modules_list = nn.ModuleList(modules)

        def forward(self, x):
            out = x
            for m in self.modules_list:
                out = m(out)
            return out


def conv_block(in_ch, out_ch, D):
    return MinkowskiSequential(
        ME.MinkowskiConvolution(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            dimension=D,
        ),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiReLU(inplace=True),
    )


def down_block(in_ch, out_ch, D):
    return MinkowskiSequential(
        ME.MinkowskiConvolution(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
            dimension=D,
        ),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiReLU(inplace=True),
    )


def up_block(in_ch, out_ch, D):
    return MinkowskiSequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
            dimension=D,
        ),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiReLU(inplace=True),
    )


class RMOSUNet(nn.Module):
    """
    Simple 3D UNet-style Minkowski network for SemanticKITTI-style voxel inputs.
    """

    def __init__(self, in_channels=1, out_channels=20, D=3, base_channels=32):
        super().__init__()
        self.D = D

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = conv_block(in_channels, c1, D)

        self.down1 = down_block(c1, c2, D)
        self.enc1 = conv_block(c2, c2, D)

        self.down2 = down_block(c2, c3, D)
        self.enc2 = conv_block(c3, c3, D)

        self.down3 = down_block(c3, c4, D)
        self.enc3 = conv_block(c4, c4, D)

        self.up2 = up_block(c4, c3, D)
        self.dec2 = conv_block(c3 + c3, c3, D)

        self.up1 = up_block(c3, c2, D)
        self.dec1 = conv_block(c2 + c2, c2, D)

        self.up0 = up_block(c2, c1, D)
        self.dec0 = conv_block(c1 + c1, c1, D)

        self.head = ME.MinkowskiConvolution(
            in_channels=c1,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dimension=D,
            bias=True,
        )

    def forward(self, x: ME.SparseTensor):
        s0 = self.stem(x)

        d1 = self.down1(s0)
        e1 = self.enc1(d1)

        d2 = self.down2(e1)
        e2 = self.enc2(d2)

        d3 = self.down3(e2)
        e3 = self.enc3(d3)

        u2 = self.up2(e3)
        u2 = ME.cat(u2, e2)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = ME.cat(u1, e1)
        u1 = self.dec1(u1)

        u0 = self.up0(u1)
        u0 = ME.cat(u0, s0)
        u0 = self.dec0(u0)

        out = self.head(u0)
        return out
