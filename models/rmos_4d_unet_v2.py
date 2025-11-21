# rmos/models/rmos_4d_unet_v2.py
import torch
import torch.nn as nn
import MinkowskiEngine as ME


def conv(in_ch, out_ch, ks, stride=1, D=4):
    return ME.MinkowskiConvolution(
        in_ch, out_ch, kernel_size=ks, stride=stride, dimension=D, bias=False
    )

def conv_tr(in_ch, out_ch, ks, stride=1, D=4):
    return ME.MinkowskiConvolutionTranspose(
        in_ch, out_ch, kernel_size=ks, stride=stride, dimension=D, bias=False
    )

def norm(ch):
    return ME.MinkowskiBatchNorm(ch)

def act():
    return ME.MinkowskiReLU(inplace=True)


class SEBlock(nn.Module):
    def __init__(self, ch, r=4, D=4):
        super().__init__()
        # ME 0.5.x pooling takes NO dimension kwarg
        self.pool = ME.MinkowskiGlobalAvgPooling()

        hidden = max(ch // r, 8)
        self.fc1 = ME.MinkowskiLinear(ch, hidden, bias=True)
        self.act = ME.MinkowskiReLU(inplace=True)
        self.fc2 = ME.MinkowskiLinear(hidden, ch, bias=True)
        self.sig = ME.MinkowskiSigmoid()

        self.mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        w = self.pool(x)     # (B, C)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.sig(w)
        return self.mul(x, w)



class ResBlock(nn.Module):
    def __init__(self, ch, D=4, se=True):
        super().__init__()
        self.conv1 = conv(ch, ch, ks=3, stride=1, D=D)
        self.bn1 = norm(ch)
        self.conv2 = conv(ch, ch, ks=3, stride=1, D=D)
        self.bn2 = norm(ch)
        self.act = act()
        self.se = SEBlock(ch, D=D) if se else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + x)


class DownBlock(nn.Module):
    """Spatial downsample only: stride=(1,2,2,2)."""
    def __init__(self, in_ch, out_ch, D=4):
        super().__init__()
        self.down = conv(in_ch, out_ch, ks=2, stride=(1,2,2,2), D=D)
        self.bn = norm(out_ch)
        self.act = act()
        self.res1 = ResBlock(out_ch, D=D)
        self.res2 = ResBlock(out_ch, D=D)

    def forward(self, x):
        x = self.act(self.bn(self.down(x)))
        x = self.res2(self.res1(x))
        return x


class UpBlock(nn.Module):
    """Spatial upsample only: stride=(1,2,2,2)."""
    def __init__(self, in_ch, skip_ch, out_ch, D=4):
        super().__init__()
        self.tconv = conv_tr(in_ch, out_ch, ks=2, stride=(1,2,2,2), D=D)
        self.bn = norm(out_ch)
        self.act = act()
        self.fuse = conv(out_ch + skip_ch, out_ch, ks=1, stride=1, D=D)
        self.res1 = ResBlock(out_ch, D=D)
        self.res2 = ResBlock(out_ch, D=D)

    def forward(self, x, skip):
        x = self.act(self.bn(self.tconv(x)))
        # cat requires same tensor_stride -> this will be true now
        x = ME.cat(x, skip)
        x = self.act(self.fuse(x))
        x = self.res2(self.res1(x))
        return x


class RMOS4DUNetV2(nn.Module):
    """
    Stronger 4D sparse UNet:
      - 4D convs
      - spatial-only down/up
      - residual + SE attention
    """
    def __init__(self, in_channels=1, num_classes=2, base_ch=48, D=4):
        super().__init__()
        self.D = D

        self.stem = nn.Sequential(
            conv(in_channels, base_ch, ks=3, stride=1, D=D),
            norm(base_ch),
            act(),
            ResBlock(base_ch, D=D),
        )

        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        self.down1 = DownBlock(c1, c2, D=D)
        self.down2 = DownBlock(c2, c3, D=D)
        self.down3 = DownBlock(c3, c4, D=D)

        self.mid = nn.Sequential(
            ResBlock(c4, D=D),
            ResBlock(c4, D=D),
        )

        self.up3 = UpBlock(c4, c3, c3, D=D)
        self.up2 = UpBlock(c3, c2, c2, D=D)
        self.up1 = UpBlock(c2, c1, c1, D=D)

        self.head = nn.Sequential(
            conv(c1, c1, ks=3, stride=1, D=D),
            norm(c1),
            act(),
            ME.MinkowskiLinear(c1, num_classes)
        )

    def forward(self, x: ME.SparseTensor):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        m = self.mid(s3)

        u3 = self.up3(m, s2)
        u2 = self.up2(u3, s1)
        u1 = self.up1(u2, s0)

        return self.head(u1)
