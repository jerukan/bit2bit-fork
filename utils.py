from spadclean import SPADHotpixelTool
from spadio import SPADData, SPADFolder
from metrics import StackMetrics, StackMetricsGroups
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import logging
import yaml
import shutil

logger = logging.getLogger(__name__)


def clean_hotpixels(input: np.ndarray) -> np.ndarray:
    clean_file = SPADHotpixelTool(input)
    return clean_file.correct_hotpixels()


def load_data(path: Path) -> np.ndarray:
    data = SPADData(path)
    return data.process(func=clean_hotpixels)


def load_directory(path: Path, use_buffer=True, reload=False) -> np.ndarray:
    processed_raw_path = path / "processed_raw_no_hotpixel.npy"
    spadfolder = SPADFolder(path)

    if processed_raw_path.exists() and use_buffer and not reload:
        logger.info(f"Loading processed data from {processed_raw_path}")
        return np.load(processed_raw_path)
    else:
        logger.info(f"Loading data from {path}")
        output = spadfolder.spadstack
        output = output.process(func=clean_hotpixels)
        if use_buffer:
            logger.info(f"Saving processed data to {processed_raw_path}")
            np.save(processed_raw_path, output.data)
            logger.info(f"Saved to {processed_raw_path}.")
        return output


def show_image(image: np.ndarray | torch.Tensor):
    """Imshow for Tensor."""
    image_sum = image.reshape((-1, image.shape[-2], image.shape[-1]))
    if isinstance(image_sum, torch.Tensor):
        image_sum = torch.sum(image_sum, dim=0).detach().cpu().numpy()
    else:
        image_sum = np.sum(image_sum, axis=0)
    plt.imshow(image_sum, cmap="gray")
    plt.title(f"Image shape: {image.shape}")
    plt.axis("off")
    plt.show()


def clear_vram():
    """Run garbage collection"""
    gc.collect()
    torch.cuda.empty_cache()
    print("Garbage collection done")


def remove_empty_directory(path: Path):
    """Remove empty directory"""
    list_dir = list(path.iterdir())
    for dir in list_dir:
        if dir.is_dir():
            remove_empty_directory(dir)
    if not list(path.iterdir()):
        path.rmdir()
        print(f"Removed empty directory: {path}")
    return path


def load_dir_path(path: Path = Path("./config.yml")):
    """Load directory path"""
    if path.exists():
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return Path(config["DATA"]["dir_path"])
    else:
        raise FileNotFoundError(f"File {path} not found.")


def load_data_path(path: Path = Path("./config.yml")):
    """Load data path"""
    if path.exists():
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return Path(config["DATA"]["data_path"])
    else:
        raise FileNotFoundError(f"File {path} not found.")


def export_metadata(path: Path, metadata: dict):
    """Export metadata to yaml file"""
    with open(path, "w") as file:
        yaml.dump(metadata, file)


def dataset_from_config(config: dict) -> SPADFolder:
    """Create SPADFolder from config"""
    data_path = Path(config["DATA"]["data_path"])
    return SPADFolder(data_path)


def list_dir(path: str | Path) -> list[Path]:
    """List directory and return list of Path objects"""
    dirctory_path = Path(path)
    dirctory_list = list(dirctory_path.iterdir())
    for i, dir in enumerate(dirctory_list):
        print(f"[{i}]:{dir}")
    return dirctory_list


def clean_directories(model_path: Path = Path("../models")):
    """Remove empty model directories"""
    model_list = [model for model in model_path.iterdir() if model.is_dir()]
    for model in model_list:
        if not list(model.glob("**/*.ckpt")):
            shutil.rmtree(model)


def group_metrics(
    input: np.ndarray,
    image: np.ndarray,
    ground_truth: np.ndarray,
    default_root_dir: Path,
    length: int = 512,
    device: torch.device = torch.device("cuda:0"),
):
    ground_truth_new = ground_truth.copy()
    for i in range(length):
        scale = np.mean(ground_truth[i]) / 255
        ground_truth_new[i] = ground_truth_new[i] / 255
        image[i] = image[i] / np.mean(image[i]) * scale
        input[i] = input[i] / np.mean(input[i]) * scale

    metric_list = ["mse", "psnr", "ssim"]
    metric1 = StackMetrics(
        image,
        ground_truth_new,
        metric_list=metric_list,
        device=device,
    )
    metric2 = StackMetrics(
        input,
        ground_truth_new,
        metric_list=metric_list,
        device=device,
    )
    metric_group = StackMetricsGroups(
        [metric1, metric2], ["processed", "raw"], metric_list
    )
    metric_group.plot_group_stats(
        save=True, save_dir=default_root_dir, save_name="group_stats"
    )
    metric_group.plot_group_trends(
        save=True,
        save_dir=default_root_dir,
        save_name="group_trends",
    )
    metric1.stats_df.to_csv(default_root_dir / "processed_stats.csv")
    metric2.stats_df.to_csv(default_root_dir / "raw_stats.csv")
    metric1.values_df.to_csv(default_root_dir / "processed_values.csv")
    metric2.values_df.to_csv(default_root_dir / "raw_values.csv")
    out = metric1.stats_df.loc["Mean", "psnr"]
    return out
