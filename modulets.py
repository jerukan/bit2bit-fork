import torch
import torch.nn as nn
import logging
from srmodules import (
    SeparableConv3d,
    PixelShuffle3d,
    PixelUnshuffle3d,
    PartialConv3d,
    PixelShuffle2d,
    PixelUnshuffle2d,
)

logger = logging.getLogger(__name__)


def conv1x1x1(
    in_channels: int,
    out_channels: int,
) -> nn.Conv3d:
    """Helper function to create 1x1x1 convolutions"""
    return nn.Conv3d(in_channels, out_channels, 1)


def conv3dnxnxn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    z_conv: bool,  # Ture = 3D convolution & False = 2D convolution
) -> nn.Conv3d:
    """Helper function to create nxnxn convolutions with padding"""
    padding_size = kernel_size // 2
    padding = padding_size if z_conv else (0, padding_size, padding_size)
    kernel = kernel_size if z_conv else (1, kernel_size, kernel_size)
    return nn.Conv3d(in_channels, out_channels, kernel, padding=padding)


def conv3x3x3(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # Ture = 3D convolution & False = 2D convolution
) -> nn.Conv3d:
    """Helper function to create 3x3x3 convolutions with padding"""
    return conv3dnxnxn(in_channels, out_channels, 3, z_conv)


def conv5x5x5(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # Ture = 3D convolution & False = 2D convolution
    separable: bool = False,
) -> nn.Conv3d | nn.Module:
    """Helper function to create 5x5x5 convolutions with padding"""
    if separable:
        return SeparableConv3d(in_channels, out_channels, 5, z_conv)
    else:
        return conv3dnxnxn(in_channels, out_channels, 5, z_conv)


def conv7x7x7(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # Ture = 3D convolution & False = 2D convolution
    separable: bool = False,
) -> nn.Conv3d | nn.Module:
    """Helper function to create 7x7x7 convolutions with padding"""
    if separable:
        return SeparableConv3d(in_channels, out_channels, 7, z_conv)
    else:
        return conv3dnxnxn(in_channels, out_channels, 7, z_conv)


def maxpool_downsample(
    z_conv: bool,  # Ture = 3D convolution & False = 2D convolution
) -> nn.MaxPool3d:
    """Helper function to create maxpooling with padding"""
    kernel_size = 2 if z_conv else (1, 2, 2)
    stride = 2 if z_conv else (1, 2, 2)
    return nn.MaxPool3d(kernel_size, stride=stride)


def avgpool_downsample(
    z_conv: bool,  # Ture = 3D convolution & False = 2D convolution
) -> nn.AvgPool3d:
    """Helper function to create avgpooling with padding"""
    kernel_size = 2 if z_conv else (1, 2, 2)
    stride = 2 if z_conv else (1, 2, 2)
    return nn.AvgPool3d(kernel_size, stride=stride)


def conv_downsample(
    in_channels: int,
    out_channels: int,
    z_conv: bool,  # Ture = 3D convolution & False = 2D convolution
) -> nn.Conv3d:
    """Helper function to create 3x3x3 convolutions with padding"""
    kernel = 3 if z_conv else (1, 3, 3)
    padding = 1 if z_conv else (0, 1, 1)
    stride = 2 if z_conv else (1, 2, 2)
    return nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding)


def pixelunshuffle(
    in_channels, out_channels, z_conv: bool, scale: int = 2
) -> nn.Module:
    return (
        nn.Sequential(
            PixelUnshuffle3d(scale),
            conv1x1x1(in_channels * (scale**3), out_channels),
        )
        if z_conv
        else nn.Sequential(
            PixelUnshuffle2d(scale),
            conv1x1x1(in_channels * (scale**2), out_channels),
        )
    )


def pool(
    in_channels: int,
    out_channels: int,
    down_mode: str,
    z_conv: bool,
    last: bool = False,
) -> nn.Module:
    if last:
        return nn.Identity()
    match down_mode:
        case "maxpool":
            return maxpool_downsample(z_conv)
        case "avgpool":
            return avgpool_downsample(z_conv)
        case "conv":
            return conv_downsample(in_channels, out_channels, z_conv)
        case "unshuffle":
            return pixelunshuffle(in_channels, out_channels, z_conv)
        case _:
            logger.warning(f"Unknown downsample mode: {down_mode}. Using maxpool.")
            return maxpool_downsample(z_conv)


def pixelshuffle(in_channels, out_channels, z_conv: bool, scale: int = 2) -> nn.Module:
    return (
        nn.Sequential(
            PixelShuffle3d(scale),
            conv3x3x3(in_channels // (scale**3), out_channels, z_conv),
        )
        if z_conv
        else nn.Sequential(
            PixelShuffle2d(scale),
            conv3x3x3(in_channels // (scale**2), out_channels, z_conv),
        )
    )


def upconv2x2x2(
    in_channels: int,
    out_channels: int,
    z_conv: bool,
    up_mode: str = "transpose",  # type of upconvolution ("transpose" | "upsample" | "pixelshuffle")
) -> nn.Module:
    """Helper function to create 2x2x2 upconvolutions"""
    kernel = 2 if z_conv else (1, 2, 2)
    stride = 2 if z_conv else (1, 2, 2)
    scale_factor = (2, 2, 2) if z_conv else (1, 2, 2)

    match up_mode:
        case "transpose":
            return nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride)
        case "pixelshuffle":
            return pixelshuffle(in_channels, out_channels, z_conv, scale=2)
        case "bilinear":
            return nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=scale_factor),
                conv3x3x3(in_channels, out_channels, z_conv),
            )
        case "nearest":
            return nn.Sequential(
                nn.Upsample(mode="nearest", scale_factor=scale_factor),
                conv3x3x3(in_channels, out_channels, z_conv),
            )
        case "trilinear":
            return nn.Sequential(
                nn.Upsample(mode="trilinear", scale_factor=scale_factor),
                conv3x3x3(in_channels, out_channels, z_conv),
            )
        case _:
            logging.warning(f"Unknown up_mode: {up_mode}. Using transpose instead.")
            return nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride)


def partial3x3x3(
    in_channels: int,
    out_channels: int,
    z_conv: bool,
    multi_channel: bool = False,
) -> PartialConv3d:
    """Helper function to create 3x3x3 partial convolutions with padding"""
    padding = 1 if z_conv else (0, 1, 1)
    kernel_size_ = (3, 3, 3) if z_conv else (1, 3, 3)
    return PartialConv3d(
        in_channels,
        out_channels,
        kernel_size_,
        padding=padding,
        multi_channel=multi_channel,
    )


def merge(
    input_a: torch.Tensor,
    input_b: torch.Tensor | None,
    merge_mode: str = "concat",
):
    """Helper function to merge two tensors"""
    if input_b is None:
        return input_a
    match merge_mode:
        case "concat":
            if input_a.shape[1:] != input_b.shape[1:]:
                logger.error(f"Unequal shape: a={input_a.shape}, b={input_b.shape}")
                raise ValueError(f"Unequal shape: a={input_a.shape}, b={input_b.shape}")
            return torch.cat((input_a, input_b), dim=1)
        case "add":
            if input_a.shape != input_b.shape:
                logger.error(f"Unequal shape: a={input_a.shape}, b={input_b.shape}")
                raise ValueError(f"Unequal shape: a={input_a.shape}, b={input_b.shape}")
            return input_a + input_b
        case _:
            logging.warning(f"Unknown merge_mode: {merge_mode}. Using concat instead.")
            return torch.cat((input_a, input_b), dim=1)


def merge_conv(
    in_channels: int,
    out_channels: int,
    z_conv: bool,
    mode: str = "concat",
) -> nn.Module:
    """Helper function to merge two tensors"""
    match mode:
        case "concat":
            return conv3x3x3(in_channels * 2, out_channels, z_conv)
        case "add":
            return conv3x3x3(in_channels, out_channels, z_conv)
        case _:
            logging.warning(f"Unknown mode: {mode}. Using concat instead.")
            return conv3x3x3(in_channels * 2, out_channels, z_conv)


def activation_function(activation: str, **kwargs) -> nn.Module:
    """Helper function to create activation layers"""
    match activation:
        case "relu":
            return nn.ReLU(inplace=kwargs.get("inplace", True))
        case "leakyrelu":
            return nn.LeakyReLU(
                kwargs.get("negative_slope", 0.01),
                inplace=kwargs.get("inplace", True),
            )
        case "prelu":
            return nn.PReLU(
                num_parameters=kwargs.get("num_parameters", 1),
                init=kwargs.get("init", 0.25),
            )
        case "gelu":
            return nn.GELU(approximate=kwargs.get("approximate", "none"))
        case "silu":
            return nn.SiLU(inplace=kwargs.get("inplace", True))
        case "tanh":
            return nn.Tanh(kwargs)
        case "sigmoid":
            return nn.Sigmoid(kwargs)
        case "softmax":
            return nn.Softmax(dim=kwargs.get("dim", None))
        case "logsoftmax":
            return nn.LogSoftmax(dim=kwargs.get("dim", None))
        case _:
            logging.warning(f"Unknown activation: {activation}. Using ReLU instead.")
            return nn.ReLU(inplace=kwargs.get("inplace", True))
