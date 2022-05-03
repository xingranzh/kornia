import torch
from torch import Tensor
from torch import cat as concatenate
from torch.nn import Conv2d, GroupNorm, LeakyReLU, Module, Sequential

from kornia.geometry import PyrDown


class DownscaleConv(Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            LeakyReLU(0.1),
            GroupNorm(8, out_channels),
            PyrDown())


class UpscaleLike(Module):
    def forward(self, x: Tensor, other: Tensor):
        return torch.nn.functional.interpolate(
            x, (other.shape[-2], other.shape[-1]),
            mode='bilinear', align_corners=False)


class UpscaleConv(Module):
    def __init__(self, in_channels: int, out_channels: int):
        self.features = Sequential(
            GroupNorm(8, in_channels),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            LeakyReLU(0.1))
        self.up = UpscaleLike()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.up(x, y)
        x = concatenate((x, y), dim=1)
        return self.features(x)


class Unet(Module):
    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 3, num_base_features: int = 32) -> None:
        super().__init__()
        self.feats_list = [2**i for i in range(num_levels)]
        # import pdb;pdb.set_trace()
        # self.stem =  nn.Sequential(
        #     nn.Conv2d(3, base_feats, kernel_size=5, stride = 1, padding=2, bias = False),
        #     nn.GroupNorm(4, base_feats),
        #     nn.LeakyReLU(0.1))
        # self.d1 = DownscaleConv(base_feats, 2*base_feats)
        # self.d2 = DownscaleConv(2*base_feats, 4*base_feats)
        # self.d3 = DownscaleConv(4*base_feats, 8*base_feats)
        # self.du3_d2short = UpscaleConv(8*base_feats, 4*base_feats, 4*base_feats)
        # self.du2_d1short = UpscaleConv(4*base_feats, 2*base_feats, 2*base_feats)
        # self.du1_d0short = UpscaleConv(2*base_feats, base_feats, base_feats)
        # self.out = Conv2d(base_feats, 3, kernel_size=1, stride = 1, padding=0, bias = True)

    def forward(self, x: Tensor) -> Tensor:
        # assert len(x.shape) == 4, x.shape
        # x = self.stem(input)
        # d1 = self.d1(x)
        # d2 = self.d2(d1)
        # d3 = self.d3(d2)
        # u3 = self.du3_d2short(d3, d2)
        # u2 = self.du2_d1short(u3, d1)
        # u1 = self.du1_d0short(u2, x)
        # out = self.out(u1) + input
        # return out
        return
