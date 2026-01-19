import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SeparableConv3d(nn.Module):
    """
    This class is a 3d version of separable convolution.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, z_conv: bool
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolving kernel
        :param z_conv: Ture = 3D convolution & False = 2D convolution
        """
        super().__init__()
        padding_size = kernel_size // 2
        padding = padding_size if z_conv else (0, padding_size, padding_size)
        kernel_size = kernel_size if z_conv else (1, kernel_size, kernel_size)  # type: ignore
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(input))


class PixelShuffle3d(nn.Module):
    """
    This class is a 3d version of pixelshuffle.
    """

    def __init__(self, scale: int = 2):
        """
        :param scale: upsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        if input.dim() != 5:
            raise ValueError(f"Input tensor must be 5D , but got {input.dim()}")
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if in_channels % (scale**3) != 0:
            raise ValueError(
                f"Input channels must be divisible by scale^3, but got {in_channels}"
            )
        out_channels = in_channels // (scale**3)
        out_z, out_x, out_y = z * scale, x * scale, y * scale
        view_shape = (batch, out_channels, scale, scale, scale, z, x, y)
        input_view = input.contiguous().view(*view_shape)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch, out_channels, out_z, out_x, out_y)


class PixelUnshuffle3d(nn.Module):
    """
    This class is a 3d version of pixelunshuffle.
    """

    def __init__(self, scale: int = 2):
        """
        :param scale: downsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        if input.dim() != 5:
            raise ValueError(f"Input tensor must be 5D , but got {input.dim()}")
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if z % self.scale != 0 or x % self.scale != 0 or y % self.scale != 0:
            raise ValueError(f"Size must be divisible by scale, but got {z}, {x}, {y}")
        out_channels = in_channels * (self.scale**3)
        out_z, out_x, out_y = z // scale, x // scale, y // scale
        view_shape = (batch, in_channels, out_z, scale, out_x, scale, out_y, scale)
        input_view = input.contiguous().view(*view_shape)
        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return output.view(batch, out_channels, out_z, out_x, out_y)


class PixelShuffle2d(nn.Module):
    """
    This class is a 2d version of pixelshuffle on BCZXY data on XY.
    """

    def __init__(self, scale: int = 2):
        """
        :param scale: upsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        if input.dim() != 5:
            raise ValueError(f"Input tensor must be 5D , but got {input.dim()}")
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if in_channels % (scale**2) != 0:
            raise ValueError(
                f"Input channels must be divisible by scale^2, but got {in_channels}"
            )
        out_channels = in_channels // (scale**2)
        out_x, out_y = x * scale, y * scale
        input_view = input.contiguous().view(batch, out_channels, scale, scale, z, x, y)
        output = input_view.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        return output.view(batch, out_channels, z, out_x, out_y)


class PixelUnshuffle2d(nn.Module):
    """
    This class is a 2d version of pixelunshuffle on BCZXY data on XY.
    """

    def __init__(self, scale: int = 2):
        """
        :param scale: downsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        if input.dim() != 5:
            raise ValueError(f"Input tensor must be 5D , but got {input.dim()}")
        scale = self.scale
        batch, in_channels, z, x, y = input.shape
        if x % self.scale != 0 or y % self.scale != 0:
            raise ValueError(f"Size must be divisible by scale, but got {x}, {y}")
        out_channels = in_channels * (self.scale**2)
        out_x, out_y = x // scale, y // scale
        view_shape = (batch, in_channels, z, out_x, scale, out_y, scale)
        input_view = input.contiguous().view(*view_shape)
        output = input_view.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        return output.view(batch, out_channels, z, out_x, out_y)


class PartialConv3d(nn.Conv3d):
    """Partial Convolutional Layer for Image Inpainting from NVIDIA."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        multi_channel: bool = False,
        return_mask: bool = False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.weight_mask_updater = (
            torch.ones(out_channels, in_channels, *self.kernel_size)
            if multi_channel
            else torch.ones(1, 1, *self.kernel_size)
        )
        mask_shape = torch.tensor(self.weight_mask_updater.shape)
        self.slide_window_size = torch.prod(mask_shape[1:])
        self.last_size = tuple()
        self.update_mask = torch.tensor([])
        self.mask_ratio = torch.tensor([])

    def forward(self, input: torch.Tensor, mask_in: torch.Tensor | None = None):
        if len(input.shape) != 5:
            logger.error(f"Input tensor must be 5D , but got {len(input.shape)}")
            raise ValueError(f"Input tensor must be 5D , but got {len(input.shape)}")
        raw_out = super().forward(torch.mul(input, mask_in) if mask_in else input)
        if mask_in or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                self.weight_mask_updater = self.weight_mask_updater.to(input)
                mask = mask_in or (
                    torch.ones(*input.data.shape).to(input)
                    if self.multi_channel
                    else torch.ones(1, 1, *input.data.shape[2:]).to(input)
                )
                self.update_mask = F.conv3d(
                    mask,
                    self.weight_mask_updater,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )
                self.mask_ratio = self.slide_window_size / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if self.bias:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        return output, self.update_mask if self.return_mask else output
