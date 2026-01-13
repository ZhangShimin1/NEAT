import logging
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch


logger = logging.getLogger(__name__)


@dataclass
class StateName:
    ADAPTER: str = "adapter"
    MODEL: str = "model"
    OPTIMIZER: str = "optimizer"
    LR_SCHEDULER: str = "lr_scheduler"
    SEED: str = "seed"
    TRAINER_STATE: str = "trainer_state"
    SCALER: str = "scaler"


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class TrainerState:
    """State of the Trainer. This can be registered by Accelerate for `save_state` and `load_state`."""

    def __init__(self, greater_is_better) -> None:
        self.epochs_trained = 0
        self.steps_trained = 0

        self.early_stopping_patience_counter = 0

        self.best_score = -np.inf if greater_is_better else np.inf
        self.best_score_epoch = 0

    def load_state_dict(self, state_dict: dict) -> None:
        self.epochs_trained = state_dict["epochs_trained"]
        self.steps_trained = state_dict["steps_trained"]

        self.early_stopping_patience_counter = state_dict[
            "early_stopping_patience_counter"
        ]

        self.best_score = state_dict["best_score"]
        self.best_score_epoch = state_dict["best_score_epoch"]

    def state_dict(self) -> dict:
        return {
            "epochs_trained": self.epochs_trained,
            "steps_trained": self.steps_trained,
            "early_stopping_patience_counter": self.early_stopping_patience_counter,
            "best_score": self.best_score,
            "best_score_epoch": self.best_score_epoch,
        }


def print_env():
    pt_version = torch.__version__
    pt_cuda_available = torch.cuda.is_available()

    info = {
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Numpy version": np.__version__,
        "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
        "System RAM": f"{psutil.virtual_memory().total / 1024**3:.2f} GB",
        "GPU Available": f"{pt_cuda_available}",
        "GPU IDs": f"{torch.cuda.device_count()}",
    }

    if pt_cuda_available:
        info["GPU type"] = torch.cuda.get_device_name()

    return "\n".join([f"- {prop}: {val}" for prop, val in info.items()])


def print_commandline_args():
    return " ".join(sys.argv)


def setup_ckpt_dir_for_epoch_i(ckpt_root_dir: Path, epoch: int):
    ckpt_epoch_dir = ckpt_root_dir / f"epoch_{str(epoch).zfill(4)}"
    ckpt_epoch_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_epoch_dir


def save_state_dict_to_checkpoint(ckpt_epoch_dir, state_dict: dict[str, Any]):
    """Save the state dictionary to the checkpoint directory.

    Note:
        For the `model` key in the state list, it can be a single model or a list of models (e.g., GAN).
        Please call this function on the rank 0 process.

    Args:
        ckpt_epoch_dir: The directory to save the checkpoint of the current epoch.
        state_dict: The state dictionary to save.
    """
    ckpt_epoch_dir.mkdir(parents=True, exist_ok=True)

    # Model states
    if StateName.MODEL in state_dict and state_dict[StateName.MODEL] is not None:
        model_states = state_dict[StateName.MODEL]

        if isinstance(model_states, list):
            for i, model_state in enumerate(model_states):
                state_fpath = ckpt_epoch_dir / f"model_{i}.pth"
                torch.save(model_state, state_fpath)
                logger.info(
                    f"Model checkpoint {os.path.getsize(state_fpath) / 1000**3:.2f} GB saved to {state_fpath}"
                )
        else:
            state_fpath = ckpt_epoch_dir / "model.pth"
            torch.save(state_dict[StateName.MODEL], state_fpath)
            logger.info(
                f"Model checkpoint {os.path.getsize(state_fpath) / 1000**3:.2f} GB saved to {state_fpath}"
            )

    # Adapter states
    if StateName.ADAPTER in state_dict and state_dict[StateName.ADAPTER] is not None:
        state_fpath = ckpt_epoch_dir / "adapter.pth"
        state_weights = state_dict[StateName.ADAPTER]
        torch.save(state_weights, state_fpath)
        logger.info(
            f"Adapter checkpoint {os.path.getsize(state_fpath) / 1000**3:.2f} GB saved to {state_fpath}"
        )

    # Optimizer states
    if (
        StateName.OPTIMIZER in state_dict
        and state_dict[StateName.OPTIMIZER] is not None
    ):
        state_fpath = ckpt_epoch_dir / "optimizer.pth"
        state_weights = state_dict[StateName.OPTIMIZER]
        torch.save(state_weights, state_fpath)
        logger.info(
            f"Optimizer checkpoint {os.path.getsize(state_fpath) / 1000**3:.2f} GB saved to {state_fpath}"
        )

    # LR Scheduler states
    if (
        StateName.LR_SCHEDULER in state_dict
        and state_dict[StateName.LR_SCHEDULER] is not None
    ):
        state_fpath = ckpt_epoch_dir / "lr_scheduler.pth"
        torch.save(state_dict[StateName.LR_SCHEDULER], state_fpath)
        logger.info(
            f"LR Scheduler checkpoint {os.path.getsize(state_fpath) / 1000**3:.2f} GB saved to {state_fpath}"
        )

    # Trainer states
    if (
        StateName.TRAINER_STATE in state_dict
        and state_dict[StateName.TRAINER_STATE] is not None
    ):
        state_fpath = ckpt_epoch_dir / "trainer_state.pkl"
        torch.save(state_dict[StateName.TRAINER_STATE], state_fpath)
        logger.info(
            f"Trainer state checkpoint {os.path.getsize(state_fpath) / 1000**3:.2f} GB saved to {state_fpath}"
        )

    if StateName.SCALER in state_dict and state_dict[StateName.SCALER] is not None:
        state_fpath = ckpt_epoch_dir / "scaler.pkl"
        torch.save(state_dict[StateName.SCALER], state_fpath)
        logger.info(
            f"Scaler checkpoint {os.path.getsize(state_fpath) / 1000**3:.2f} GB saved to {state_fpath}"
        )


def load_state_dict_from_checkpoint(ckpt_epoch_dir: Path) -> dict[str, Any]:
    """Load all possible states from the checkpoint directory (e.g., model, adapter, optimizer, lr_scheduler, trainer).

    Note:
        For the `model` key in the state list, it can be a single model or a dictionary of models (e.g., GAN).
        Please call this function on the rank 0 process.

    Args:
        ckpt_epoch_dir: The directory to load the checkpoint of the current epoch.

    Returns:
        The state dictionary loaded from the checkpoint directory.
    """
    state_dict = {}

    # Model states
    model_state_fpaths = list(ckpt_epoch_dir.glob("model*.pth"))

    if len(model_state_fpaths) > 1:
        model_states = {}
        for model_state_fpath in model_state_fpaths:
            model_name = model_state_fpath.stem
            model_states[model_name] = torch.load(
                model_state_fpath, map_location="cpu", weights_only=True
            )
        state_dict[StateName.MODEL] = model_states
    elif len(model_state_fpaths) == 1:
        state_dict[StateName.MODEL] = torch.load(
            model_state_fpaths[0], map_location="cpu", weights_only=True
        )

    # Adapter states
    adapter_state_fpath = ckpt_epoch_dir / "adapter.pth"
    if adapter_state_fpath.exists():
        state_dict[StateName.ADAPTER] = torch.load(
            adapter_state_fpath, map_location="cpu", weights_only=True
        )

    # Optimizer states
    optimizer_state_fpath = ckpt_epoch_dir / "optimizer.pth"
    if optimizer_state_fpath.exists():
        state_dict[StateName.OPTIMIZER] = torch.load(
            optimizer_state_fpath, map_location="cpu", weights_only=True
        )

    # LR Scheduler states
    lr_scheduler_state_fpath = ckpt_epoch_dir / "lr_scheduler.pth"
    if lr_scheduler_state_fpath.exists():
        state_dict[StateName.LR_SCHEDULER] = torch.load(
            lr_scheduler_state_fpath, map_location="cpu", weights_only=True
        )

    # Trainer states
    trainer_state_fpath = ckpt_epoch_dir / "trainer_state.pkl"
    if trainer_state_fpath.exists():
        state_dict[StateName.TRAINER_STATE] = torch.load(
            trainer_state_fpath, weights_only=False
        )

    # Scaler states
    scaler_state_fpath = ckpt_epoch_dir / "scaler.pkl"
    if scaler_state_fpath.exists():
        state_dict[StateName.SCALER] = torch.load(scaler_state_fpath, weights_only=True)

    return state_dict


def cleanup_redundant_checkpoints(ckpt_root_dir: Path, num_keep_ckpts: int):
    ckpt_epoch_dirs = sorted(
        ckpt_root_dir.glob("epoch_*"),
        key=lambda x: int(x.stem.split("_")[-1]),
        reverse=True,
    )

    for ckpt_epoch_dir in ckpt_epoch_dirs[num_keep_ckpts:]:
        shutil.rmtree(ckpt_epoch_dir)
        logger.info(f"Checkpoint {ckpt_epoch_dir} is removed.")

    logger.info(f"Kept the latest {num_keep_ckpts} checkpoints.")
