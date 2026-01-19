from dataclasses import dataclass, field
import json
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import logging
from typing import ClassVar
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainData:
    data: np.ndarray | tuple[np.ndarray, np.ndarray]
    xy_size: int
    z_size: int
    min_psnr: float | str = "auto"
    max_psnr: float | str = "auto"
    virtual_size: int = 0
    psnr_sampling: str = "db"
    augments: bool = False
    rotation: float = 0
    random_crop: bool = True
    skip_frames: int = 1
    max_probability: float = 0.99
    ground_truth: np.ndarray | None = None
    note: str = ""

    _frame_psnr: np.ndarray | None = None
    _MAX_PSNR_STACK_SIZE: ClassVar[int] = 1000

    @property
    def name(self) -> str:
        return f"{self.note}_{self.stacksize}x{self.z_size}x{self.xy_size}x{self.xy_size}_skip={self.skip_frames}"

    @property
    def stacksize(self) -> int:
        return (
            self.virtual_size
            if self.virtual_size > 0
            else len(self.data if isinstance(self.data, np.ndarray) else self.data[0])
        )

    @property
    def psnr_skip_frames(self) -> int:
        data_len = len(self.data if isinstance(self.data, np.ndarray) else self.data[0])
        if data_len < self._MAX_PSNR_STACK_SIZE:
            return 1
        else:
            return data_len // self._MAX_PSNR_STACK_SIZE

    @property
    def frame_psnr(self) -> np.ndarray:
        if self._frame_psnr is None:
            input = (
                self.data[:: self.psnr_skip_frames]
                if isinstance(self.data, np.ndarray)
                else self.data[0][:: self.psnr_skip_frames]
            )
            self._frame_psnr = np.mean(input, axis=(1, 2))
        return self._frame_psnr

    @property
    def mean_psnr(self) -> float:
        return np.mean(self.frame_psnr).item()

    @property
    def _min_psnr(self) -> float:
        if isinstance(self.min_psnr, float | int):
            return self.min_psnr
        else:
            return 10 * np.log10(np.min(self.frame_psnr)).item()

    @property
    def _max_psnr(self) -> float:
        if isinstance(self.max_psnr, float | int):
            return self.max_psnr
        else:
            return 10 * np.log10(np.max(self.frame_psnr)).item()

    @property
    def each_frame_psnr(self) -> np.ndarray:
        return 10 * np.log10(self.mean_psnr)

    @property
    def sample_size_in_GB(self) -> float:
        return (self.xy_size**2) * self.z_size * 4 / (1024**3)

    def metadata(self):
        return {
            "raw_data_shape": f"{self.data.shape if isinstance(self.data, np.ndarray) else self.data[0].shape}",
            "xy_size": self.xy_size,
            "z_size": self.z_size,
            "min_psnr": self._min_psnr,
            "max_psnr": self._max_psnr,
            "stacksize": self.stacksize,
            "virtual_size": self.virtual_size,
            "psnr_sampling": self.psnr_sampling,
            "random_crop": self.random_crop,
            "augments": self.augments,
            "rotation": self.rotation,
            "max_probability": self.max_probability,
            "note": self.note,
            "mean_psnr": self.mean_psnr,
            "data_str": self.name,
            "sample_size_in_GB": self.sample_size_in_GB,
        }

    @classmethod
    def from_config(
        cls,
        config: dict,
        data: np.ndarray | tuple[np.ndarray, np.ndarray],
        ground_truth: np.ndarray | None = None,
    ) -> "TrainData":
        return cls(
            data=data,
            ground_truth=ground_truth,
            xy_size=config["xy_size"],
            z_size=config["z_size"],
            min_psnr=config["min_psnr"],
            max_psnr=config["max_psnr"],
            virtual_size=config["virtual_size"],
            psnr_sampling=config["psnr_sampling"],
            random_crop=config["random_crop"],
            augments=config["augments"],
            rotation=config["rotation"],
            max_probability=config["max_probability"],
            skip_frames=config["skip_frames"],
            note=config["note"],
        )

    @classmethod
    def from_config_validation(
        cls,
        config: dict,
        data: np.ndarray | tuple[np.ndarray, np.ndarray],
        ground_truth: np.ndarray | None = None,
    ) -> "TrainData":
        return cls(
            data=data,
            ground_truth=ground_truth,
            xy_size=config["xy_size"],
            z_size=config["z_size"],
            min_psnr=config["min_psnr"],
            max_psnr=config["max_psnr"],
            virtual_size=0,
            psnr_sampling="fixed",
            random_crop=False,
            augments=False,
            rotation=False,
            max_probability=1,
            skip_frames=0,
            note=config["note"],
        )

    def __call__(self):
        return self.data


@dataclass
class ModelConfig:
    channels: int = 1
    depth: int = 4
    start_filts: int = 24
    depth_scale: int = 2
    depth_scale_stop: int = 10
    z_conv_stage: int = 5
    group_norm: int = 0
    skip_depth: int = 0
    dropout_p: float = 0.0
    scale_factor: float = 10.0
    sin_encoding: bool = True
    signal_levels: int = 10
    masked: bool = True
    down_checkpointing: bool = False
    up_checkpointing: bool = False
    loss_function: str = "photon"
    up_mode: str = "transpose"
    merge_mode: str = "concat"
    down_mode: str = "maxpool"
    activation: str = "relu"
    block_type: str = "dual"
    note: str = ""
    _optimizer_config: dict = field(init=False, default_factory=dict)

    @property
    def name(self) -> str:
        name_str = [
            f"l={self.signal_levels}",
            f"d={self.depth}",
            f"sf={self.start_filts}",
            f"ds={self.depth_scale}at{self.depth_scale_stop}",
            f"f={self.scale_factor}",
            f"z={self.z_conv_stage}",
            f"g={self.group_norm}",
            f"sd={self.skip_depth}",
            f"b={self.block_type}",
            f"a={self.activation}",
        ]
        return make_name(name_str)

    @property
    def optimizer_config(self) -> dict:
        return self._optimizer_config

    @optimizer_config.setter
    def optimizer_config(self, value: dict):
        self._optimizer_config = value

    @classmethod
    def from_config(cls, config: dict) -> "ModelConfig":
        instance = cls(
            channels=config["channels"],
            depth=config["depth"],
            start_filts=config["start_filts"],
            depth_scale=config["depth_scale"],
            depth_scale_stop=config["depth_scale_stop"],
            z_conv_stage=config["z_conv_stage"],
            group_norm=config["group_norm"],
            skip_depth=config["skip_depth"],
            dropout_p=config["dropout_p"],
            scale_factor=config["scale_factor"],
            sin_encoding=config["sin_encoding"],
            signal_levels=config["signal_levels"],
            masked=config["masked"],
            down_checkpointing=config["down_checkpointing"],
            up_checkpointing=config["up_checkpointing"],
            loss_function=config["loss_function"],
            up_mode=config["up_mode"],
            merge_mode=config["merge_mode"],
            down_mode=config["down_mode"],
            activation=config["activation"],
            block_type=config["block_type"],
            note=config["note"],
        )
        instance.optimizer_config = config["optimizer_config"]
        return instance

    def metadata(self) -> dict:
        return {
            "channels": self.channels,
            "depth": self.depth,
            "start_filts": self.start_filts,
            "depth_scale": self.depth_scale,
            "depth_scale_stop": self.depth_scale_stop,
            "skip_depth": self.skip_depth,
            "z_conv_stage": self.z_conv_stage,
            "group_norm": self.group_norm,
            "dropout_p": self.dropout_p,
            "scale_factor": self.scale_factor,
            "sin_encoding": self.sin_encoding,
            "signal_levels": self.signal_levels,
            "masked": self.masked,
            "down_checkpointing": self.down_checkpointing,
            "up_checkpointing": self.up_checkpointing,
            "loss_function": self.loss_function,
            "up_mode": self.up_mode,
            "merge_mode": self.merge_mode,
            "down_mode": self.down_mode,
            "activation": self.activation,
            "block_type": self.block_type,
            "note": self.note,
            "model_name": self.name,
            "optimizer_config": self.optimizer_config,
        }


@dataclass
class TrainConfig:
    data: TrainData
    model: ModelConfig
    batch_size: int = 100
    epochs: int = 50
    shuffle: bool = False
    drop_last: bool = True
    pin_memory: bool = True
    num_workers: int = 0
    device_number: int = 0
    precision: str | int = 32
    matmul_precision: str = "high"
    note: str = ""

    @property
    def name(self) -> str:
        name_str = [
            f"{self.time_stamp}",
            f"{self.data.name}",
            f"{self.model.name}",
            f"b={self.batch_size}",
            f"e={self.epochs}",
            f"p={self.precision}",
            f"n={self.note}" if self.note != "" else None,
        ]
        return make_name(name_str)

    @property
    def time_stamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def device(self) -> torch.device:
        try:
            output = torch.device(f"cuda:{self.device_number}")
        except RuntimeError:
            raise ValueError("No GPU available")
        return output

    @property
    def iterations_per_epoch(self) -> int:
        length = self.data.stacksize
        return length // self.batch_size

    @property
    def estimate_unet_vram(self) -> float:
        filters_at_depth = [
            self.model.start_filts * (2**i) for i in range(self.model.depth)
        ]
        total_params = 0
        for i in range(self.model.depth):
            in_channels = filters_at_depth[i] if i == 0 else filters_at_depth[i - 1]
            out_channels = filters_at_depth[i]
            params_per_conv = (in_channels * out_channels * 3**3) + out_channels
            total_params += 2 * params_per_conv
        total_params += (filters_at_depth[-1] * 1 * 1**3) + 1
        memory_params = total_params * 4
        xy_spatial_size = [self.data.xy_size // (2**i) for i in range(self.model.depth)]
        z_spatial_size = [self.data.z_size // (2**i) for i in range(self.model.depth)]
        memory_features = []
        for i in range(self.model.depth):
            if i == 0:
                memory_features.append(
                    self.batch_size
                    * filters_at_depth[i]
                    * (self.data.xy_size**2)
                    * self.data.z_size
                    * 4
                )
            else:
                memory_features.append(
                    self.batch_size
                    * filters_at_depth[i]
                    * (xy_spatial_size[i] ** 2)
                    * z_spatial_size[i]
                    * 4
                )
                memory_features.append(
                    self.batch_size
                    * filters_at_depth[i - 1]
                    * (xy_spatial_size[i] ** 2)
                    * z_spatial_size[i]
                    * 4
                )
        memory_features = sum(memory_features)
        total_vram = memory_params + memory_features
        total_vram_gb = total_vram / (1024 * 1024 * 1024)
        return total_vram_gb

    def metadata(self) -> dict:
        return {
            "data": self.data.metadata(),
            "model": self.model.metadata(),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
            "device_number": self.device_number,
            "device": str(self.device),
            "precision": self.precision,
            "matmul_precision": self.matmul_precision,
            "note": self.note,
            "time_stamp": self.time_stamp,
            "iterations_per_epoch": self.iterations_per_epoch,
            "estimate_unet_vram": self.estimate_unet_vram,
            "estimate_data_vram": self.data.sample_size_in_GB * self.batch_size,
        }

    def to_json(self, path: Path | None = None) -> str:
        json_output = json.dumps(self.metadata(), indent=4)
        if path is not None:
            with open(path, "w") as f:
                f.write(json_output)
        return json_output

    def to_yaml(self, path: Path | None = None) -> str:
        yaml_output = yaml.dump(
            self.metadata(),
            default_flow_style=False,
            sort_keys=False,
            encoding=None,
            indent=2,
        )
        if path is not None:
            with open(path, "w") as f:
                f.write(yaml_output)
        return yaml_output

    @classmethod
    def from_config(
        cls, config: dict, data: TrainData, model: ModelConfig
    ) -> "TrainConfig":
        return cls(
            data=data,
            model=model,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            shuffle=config["shuffle"],
            drop_last=config["drop_last"],
            pin_memory=config["pin_memory"],
            num_workers=config["num_workers"],
            device_number=config["device_number"],
            precision=config["precision"],
            matmul_precision=config["matmul_precision"],
            note=config["note"],
        )


def load_config(path: Path = Path("./config.yml")) -> dict:
    """Load yaml configuration file"""
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def make_name(name_list: list[str]) -> str:
    """Join a list of strings into a single string"""
    return "_".join([name for name in name_list if name is not None and name != ""])
