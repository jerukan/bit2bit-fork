# import torch, traceback
# orig = getattr(torch._C, "_cuda_init", None)
# def _wrapped_cuda_init():
#     print("=== torch._C._cuda_init called ===")
#     traceback.print_stack(limit=8)
#     if orig:
#         orig()
# torch._C._cuda_init = _wrapped_cuda_init


from spadio import SPADFolder, SPADData  # noqa
from spadclean import GenerateTestData, SPADHotpixelTool  # noqa
from pathlib import Path
from utils import clean_hotpixels
from inference import cpu_inference
from metadata import TrainData, ModelConfig, TrainConfig, load_config
from dataset import (
    BernoulliDataset3D,
    ValidationDataset3D,
    PairedDataset,
    BinomDataset3D,
    N2NDataset3D,
)  # noqa
from spadgapmodels import SPADGAP
import torch
import torch.utils.data as dt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from tifffile import imwrite, imread
import numpy as np
import logging
import sys
import shutil
from tqdm.auto import tqdm
import dask.array as da
import zarr
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

from inference import gpu_patch_inference

from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
def get_codec_for_format(format: str):
    """
    Get appropriate fourcc codec string for given video format.
    """
    format = format.lower()
    if format == "mp4":
        return "mp4v"
    elif format == "avi":
        return "FFV1"
    elif format == "mov":
        return "avc1"
    else:
        raise ValueError(f"I haven't added the codec for: {format}")

def to_video(
    frames: np.ndarray, path, res_scale=1.0, playback_fps=None, gamma=1.0, cmap=None, format=None,
    vmin=None, vmax=None, use_quantile=False
):
    """
    Saves video frame arrays to a video file or sequence of PNGs. If path has no extension, 
    it is treated as a directory and individual image files are saved.

    Args:
        frames (np.ndarray): (T x H x W x C) (RGB) or (T x H x W) (intensity) video frames.
        path (str or Path): output video file path or directory for image files.
        res_scale (float): resolution scaling factor with nearest neighbor interpolation.
        cmap: ignored if frames are RGB; otherwise, matplotlib colormap name or object.
        format (str or None): video format (e.g., "mp4", "avi"), or image format (e.g., "png");
            if None, inferred from path suffix.
        use_quantile (bool): if True, use quantiles to determine vmin and vmax for normalization
            (ignored if vmin or vmax are specified).
    """
    path = Path(path)
    if cmap is None:
        cmap = "viridis"
    cmap_fn = plt.get_cmap(cmap)
    is_rgb = False
    if frames.ndim == 4:
        if frames.shape[3] == 3:
            is_rgb = True
        else:
            raise ValueError("4D frames array must have shape (T, H, W, 3) for RGB video")
    elif frames.ndim == 3:
        is_rgb = False
    else:
        raise ValueError("frames must be a 3D or 4D numpy array")

    # compute a normalized intensity in [0,1] for colormap input
    if vmax is None:
        if use_quantile:
            vmax = float(np.quantile(frames, 0.99))
        else:
            vmax = float(np.max(frames))
    if vmin is None:
        if use_quantile:
            vmin = float(np.quantile(frames, 0.01))
        else:
            vmin = float(np.min(frames))

    H, W = frames.shape[1], frames.shape[2]
    if res_scale != 1.0:
        out_W = int(W * res_scale)
        out_H = int(H * res_scale)
    else:
        out_W = W
        out_H = H
    # if path is a directory, write individual image files
    is_video_file = path.suffix in [".mp4", ".avi", ".mov", ".mkv"]
    if not is_video_file:
        path.mkdir(parents=True, exist_ok=True)
        if format is None:
            format = "png"
    else:
        if playback_fps is None:
            raise ValueError("playback_fps must be specified if saving a video file")
        path.parent.mkdir(parents=True, exist_ok=True)
        if format is None:
            format = path.suffix[1:].lower()
        codec = get_codec_for_format(format)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vidwriter = cv2.VideoWriter(str(path), fourcc, playback_fps, (out_W, out_H), isColor=True)

    max_frames = len(frames)

    if not is_video_file:
        allpaths = []
    for i in tqdm(range(max_frames), desc="Writing video frames"):
        intensity = (np.clip(frames[i], vmin, vmax) - vmin) / (vmax - vmin)  # normalize to [0,1]
        if gamma != 1:
            intensity = intensity ** gamma
        if is_rgb:
            rgb_mapped = (intensity * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        else:
            # apply matplotlib colormap -> returns RGBA in [0,1]
            rgba_mapped = cmap_fn(intensity)  # shape (H,W,4)
            rgb_mapped = (rgba_mapped[..., :3] * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        bgr_mapped = rgb_mapped[..., ::-1]  # convert to BGR for OpenCV
        if res_scale != 1.0:
            bgr_mapped = cv2.resize(bgr_mapped, (out_W, out_H), interpolation=cv2.INTER_NEAREST)
        if is_video_file:
            vidwriter.write(bgr_mapped)
        else:
            frame_path = path / f"frame_{i:05d}.{format}"
            cv2.imwrite(str(frame_path), bgr_mapped)
            allpaths.append(frame_path)
    if is_video_file:
        vidwriter.release()
    if not is_video_file:
        return allpaths
    return path

def read_quanta_zarr(path, load_data=True):
    """
    Reads quanta data from a Zarr file and returns points and quantaframes.

    Args:
        path (str or Path): Path to the Zarr file.
        load_data (bool): if True, loads all arrays into memory.

    Returns:
        tuple:
            - quantaframes: array of shape (T, H, W) with binary frames. If
                load_data is False, the array will be lazy-loaded.
            - (t, y, x) (np.ndarray): tuple of 1D arrays for t, y, and x coordinates.
            - attrs (dict): relevant metadata (fps and T_exp).
    """
    zarrdata = zarr.open_group(path, mode="r")
    framekey = "frames" if "frames" in zarrdata else "quantaframes"
    contains_coords = all(k in zarrdata for k in ("t", "y", "x"))
    quantaframes = zarrdata[framekey]
    if contains_coords:
        t = zarrdata["t"]
        y = zarrdata["y"]
        x = zarrdata["x"]
    else:
        t = y = x = None
    if load_data:
        quantaframes = quantaframes[:]
        if contains_coords:
            t = t[:]
            y = y[:]
            x = x[:]
    return quantaframes, (t, y, x), zarrdata.attrs

def prob_from_dcr(dcr_rate_hz, fps):
    """
    Convert a dark count rate in Hz to a per-frame probability of a dark count photon.
    """
    return 1 - np.exp(-dcr_rate_hz / fps)

def thin_frames_uniform(frames, keep_prob, dcr_prob=None, seed=None):
    """
    Thin binary frames uniformly with probability keep_prob. Also adds dark
    count photons to lower the SNR.

    This is an expensive operation on SPAD data, so dask is used for
    multiprocessing.

    Args:
        dcr_prob: dark count photon probability (not the rate itself)
    """
    T, H, W = frames.shape
    # convert to a dask array with automatic chunking and apply a lazy random mask
    frames = da.from_array(frames, chunks=(400, H, W))
    rs = da.random.RandomState(seed)
    mask = rs.random_sample(frames.shape, chunks=frames.chunks) < keep_prob
    if dcr_prob is not None:
        dcr_photons = rs.binomial(1, dcr_prob, size=frames.shape, chunks=frames.chunks).astype("uint8")
    else:
        dcr_photons = da.zeros(frames.shape, chunks=frames.chunks, dtype="uint8")
    frames = (frames.astype("uint8") & mask.astype("uint8")) | dcr_photons
    return frames.compute()

configure_path = Path("./config.yml")
config = load_config(path=configure_path)  # CLI argument

datanames = ["Monkey", "cpufan_restarget", "Resolution_target_drill", "ultrasound_bubble24", "plasma_ball_5_med"]
keep_probs = [1.0, 1/32]

for dataname in datanames:
    data_type = config["PATH"]["data_type"]
    if data_type not in ["raw", "processed", "zarr"]:
        raise ValueError("Data type must be RAW or CLEAN or ZARR")

    # data_path = Path(config["PATH"]["data_path"])
    data_path = Path("/scratch/jeyan/photon-testing-grounds/data/quanta") / f"{dataname}.zarr"
    num_of_files = config["PATH"]["num_of_files"]
    data_file = config["PATH"]["data_file"]
    model_path = Path(config["PATH"]["model_path"])

    if data_type == "raw":
        try:
            if data_path.is_dir():
                input_folder = SPADFolder(data_path)
                input = input_folder.spadstack[:num_of_files]
                data_orig = input.process(clean_hotpixels)
            else:
                input_folder = SPADData(data_path)
                input = input_folder.data
                data_orig = clean_hotpixels(input)
        except FileNotFoundError:
            logging.error("Folder not found")
            # sys.exit(1)
        del input
    elif data_type == "zarr":
        data_orig, _, _ = read_quanta_zarr(data_path)
    FRAME_LIMIT = 40000
    data_orig = data_orig[:FRAME_LIMIT]

    for keep_prob in keep_probs:
        # keep_prob = config["PATH"]["thin"]
        if keep_prob < 1.0:
            dcr_prob = prob_from_dcr(dcr_rate_hz=25, fps=100000)
            data = thin_frames_uniform(data_orig, keep_prob=keep_prob, dcr_prob=dcr_prob, seed=42)
        else:
            data = data_orig

        dataset = f"{data_path.stem}-thin{keep_prob:.3f}"
        model = SPADGAP.load_from_checkpoint(f"models/{dataset}/final_model.ckpt")
        indata = data[:10000].astype(np.float32)
        output = gpu_patch_inference(
            model,
            indata,
            initial_patch_depth=48,
            min_overlap=40,
            device=1,
        )
        resdir = Path(f"results/{dataset}")
        resdir.mkdir(parents=True, exist_ok=True)
        compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
        zarr.create_array(
            store=resdir / "inference.zarr",
            data=output,
            overwrite=True,
            compressors=compressor,
            chunks=(400, 512, 512),
        )
        to_video(output, resdir / "inference-gamma.mp4",  playback_fps=30, gamma=1/2.2, cmap="grey", vmin=0)
