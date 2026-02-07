import logging
import math
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchinfo import summary
from tqdm import tqdm

from neat.src.accelerate import (
    gather_object,
    get_world_size_and_rank,
    wait_for_everyone,
)
from neat.src.logger import TensorboardLogger
from neat.src.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from neat.src.trainer_args import TrainingArgs
from neat.src.trainer_utils import (
    StateName,
    TrainerState,
    load_state_dict_from_checkpoint,
    print_commandline_args,
    print_env,
    save_state_dict_to_checkpoint,
)
from neat.src.utils import (
    cleanup_before_training,
    prepare_empty_dir,
    set_random_seed,
)


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArgs,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        **kwargs,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: The model to train, evaluate, or test.
            args: The training arguments. This is a dataclass that contains all the training arguments.
            data_collator: The data collator to use. Defaults to None.
            train_dataset: The training dataset. Defaults to None.
            eval_dataset: The evaluation dataset. Defaults to None. If can be a single dataset or a dictionary of datasets.
            optimizers: The optimizer and the learning rate scheduler. Defaults to (None, None).

        Important attributes:
            **in_training**: Whether or not a model is currently running `train` (e.g. when `evaluate` is called while in `train`)
        """
        self.in_training = False
        # Set a flag tensor for early stopping and other breakpoints
        self.flag_tensor = None

        self.args = args

        # Set device as early as possible
        self.world_size, self.rank = get_world_size_and_rank()
        self.is_rank_zero = self.rank == 0

        # Set random seed
        self.seed = set_random_seed(seed=args.seed)

        # Datasets, model, optimizer, lr_scheduler
        self.data_collator = data_collator
        self.train_dataset = train_dataset

        # Dataloaders. They will be created in the `train` and `evaluate` methods.
        self.train_dataloader = None
        self.eval_dataloaders = None

        self.eval_dataset = eval_dataset
        self.model = model
        self.optimizer, self.lr_scheduler = optimizers

        # Setup directories
        self._setup_exp_paths(output_dir=args.output_dir)

        # Acoustic parameters
        self._inject_acoustic_args()

        # Trainer states
        self.state = TrainerState(greater_is_better=self.args.greater_is_better)

        # Pandas settings for better display
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        if self.is_rank_zero:
            prepare_empty_dir(
                [
                    self.checkpoints_dir,
                    self.tb_log_dir,
                    self.enhanced_dir,
                    self.metrics_dir,
                ],
                resume=self.args.resume_from_checkpoint != "no",
            )

            logger.info(f"\nEnvironment information:\n{print_env()}")

            logger.info("===== Training Arguments =====")
            logger.info(print_commandline_args())
            logger.info("=" * 30)

            self.writer = TensorboardLogger(self.tb_log_dir.as_posix())
            self.writer.log_config(self.args.to_dict())

            # Model summary
            logger.info(f"\n{summary(self.model, verbose=0)}")

        wait_for_everyone()

    @staticmethod
    def _search_latest_ckpt_path(checkpoints_dir: Path) -> Path:
        """Find the latest checkpoint path from a given checkpoint directory."""
        # Pick up all checkpoints with the format `epoch_*`
        checkpoints = sorted(checkpoints_dir.glob("epoch_" + ("[0-9]" * 4)))

        # Remove files that is not a checkpoint
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.is_dir()]

        if len(checkpoints) == 0:
            raise FileNotFoundError(
                f"No checkpoints found in {checkpoints_dir.as_posix()}."
            )

        # Pick up the latest checkpoint
        ckpt_path = checkpoints[-1]

        return ckpt_path

    def _parse_ckpt_path_or_alias(self, path_or_alias: str) -> Optional[Path]:
        if (
            path_or_alias not in ["no", "latest", "best"]
            and not Path(path_or_alias).exists()
        ):
            raise FileNotFoundError(f"Checkpoint not found in {path_or_alias}")

        ckpt_path = None

        if path_or_alias == "no":
            if self.is_rank_zero:
                logger.info("Training from scratch. No checkpoint is loaded.")
            return ckpt_path
        elif path_or_alias == "latest":
            ckpt_path = self._search_latest_ckpt_path(self.checkpoints_dir)
        elif path_or_alias == "best":
            ckpt_path = self.checkpoints_dir / "best"
        else:
            ckpt_path = Path(path_or_alias).expanduser().resolve()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found in {ckpt_path.as_posix()}")

        return ckpt_path

    def _load_state_dict_from_checkpoint(self, path_or_alias: str) -> Dict[str, Any]:
        """Load the state dictionary from the checkpoint directory."""
        if (
            path_or_alias not in ["no", "latest", "best"]
            and not Path(path_or_alias).exists()
        ):
            raise FileNotFoundError(f"Checkpoint not found in {path_or_alias}")

        state_dict: Dict[str:Any] = {}

        # Parse the checkpoint path or alias
        if path_or_alias == "no":
            if self.is_rank_zero:
                logger.info("Training from scratch. No checkpoint is loaded.")
            return state_dict
        elif path_or_alias == "latest":
            ckpt_path = self._search_latest_ckpt_path(self.checkpoints_dir)
        elif path_or_alias == "best":
            ckpt_path = self.checkpoints_dir / "best"
        else:
            ckpt_path = Path(path_or_alias).expanduser().resolve()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found in {ckpt_path.as_posix()}")

        # Load the checkpoint
        return load_state_dict_from_checkpoint(ckpt_path)

    def save_checkpoint(self, epoch: int, is_best_epoch: bool = False):
        """Save the checkpoint.

        Args:
            epoch: The current epoch.
            is_best_epoch: Whether the current epoch is the best epoch. Defaults to False.
        """
        if is_best_epoch:
            ckpt_path = self.checkpoints_dir / "best"
        else:
            ckpt_path = self.checkpoints_dir / f"epoch_{str(epoch).zfill(4)}"

        if isinstance(self.model, DDP):
            logger.info(
                "Extract the `state_dict` from the `model.module.state_dict()`..."
            )
            model_stain_dict = self.model.module.state_dict()
        else:
            logger.info("Extract the `state_dict` from the `model.state_dict()`...")
            model_stain_dict = self.model.state_dict()

        state_dict = {StateName.MODEL: model_stain_dict}
        state_dict[StateName.OPTIMIZER] = self.optimizer.state_dict()
        state_dict[StateName.LR_SCHEDULER] = self.lr_scheduler.state_dict()
        state_dict[StateName.TRAINER_STATE] = self.state.state_dict()

        # Find all checkpoints
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_" + ("[0-9]" * 4)))
        if epoch < len(checkpoints):
            logger.warning(
                f"Current epoch is {epoch}, but found {len(checkpoints)} checkpoints. "
                f"This may be caused by you running the same experiment multiple times. "
                f"Recommend to run the experiment with a different `output_dir`."
                f"Otherwise, the newer checkpoints will be removed."
            )

        # Remove the old checkpoints
        if len(checkpoints) > self.args.save_total_limit:
            logger.info(
                f"Found {len(checkpoints)} checkpoints, keeping the latest {self.args.save_total_limit} checkpoints."
            )

            for checkpoint_dir in checkpoints[: -self.args.save_total_limit]:
                shutil.rmtree(checkpoint_dir.as_posix())
                logger.info(f"Checkpoint {checkpoint_dir.as_posix()} is removed.")

        save_state_dict_to_checkpoint(ckpt_path, state_dict)

    def _prepare_ddp_model(self, model):
        """Prepare the model for acceleration.

        Notes:
            You should never try to change your model's parameters after wrapping up your model with DDP. Because,
            when wrapping up your model with DDP, the constructor of DDP will register the additional gradient reduction
            functions on all the parameters of the model itself at time of construction. If you change the parameters of
            the model after wrapping up the model with DDP, the gradient reduction functions will not be matched with the
            correct parameters of the model. It means that you must load the model's weights before wrapping up the model
            with DDP. See: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

        Args:
            model: The model to prepare.

        Returns:
            The model prepared for acceleration.
        """
        if dist.is_available() and dist.is_initialized():
            model = model.to(self.rank)
            model = DDP(
                model, find_unused_parameters=self.args.ddp_find_unused_parameters
            )
            # synchronize before training begins
            dist.barrier()
            return model
        else:
            model = model.to("cpu")
            return model

    @staticmethod
    def _current_time_now():
        return time.strftime("%Y_%m_%d--%H_%M_%S")

    def _setup_exp_paths(self, output_dir: str):
        """Set the paths for the experiment.

        Args:
            output_dir: the root directory to save all experiments.

        Example:
            - output_dir: /home/xhao/exp/fullsubnet_lr_0.g
            - checkpoints_dir: /home/xhao/exp/fullsubnet_lr_0.1/checkpoints
            - tb_log_dir: /home/xhao/exp/fullsubnet_lr_0.1/tb_log
            - enhanced_dir: /home/xhao/exp/fullsubnet_lr_0.1/enhanced
            - metrics_dir: /home/xhao/exp/fullsubnet_lr_0.1/metrics
            - source_code_backup_dir: /home/xhao/exp/fullsubnet_lr_0.1/source_code__YYYY_MM_DD__HH_MM_SS
            - model_args_path: /home/xhao/exp/fullsubnet_lr_0.1/model_args__YYYY_MM_DD__HH_MM_SS.yaml
        """
        time_now = self._current_time_now()  # returns a timestamp string

        self.output_dir = Path(output_dir).expanduser().absolute()
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.tb_log_dir = self.output_dir / "tb_log"
        self.enhanced_dir = self.output_dir / "enhanced"
        self.metrics_dir = self.output_dir / "metrics"

        # Each run will have a unique source code, config, and log file.
        self.source_code_dir = (
            Path(__file__).expanduser().absolute().parent.parent.parent
        )
        self.source_code_backup_dir = self.output_dir / f"source_code__{time_now}"
        self.model_args_path = self.output_dir / f"model_args__{time_now}.yaml"
        self.loss_log_path = self.output_dir / "loss.csv"

    def _inject_acoustic_args(self):
        """Setup acoustic arguments."""
        n_fft = self.args.acoustic_n_fft
        hop_length = self.args.acoustic_hop_length
        win_length = self.args.acoustic_win_length
        sr = self.args.acoustic_sr

        # Support for torch and librosa stft
        self.torch_stft = partial(
            torch.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.torch_istft = partial(
            torch.istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.librosa_stft = partial(
            librosa.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.librosa_istft = partial(
            librosa.istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sr = sr

    def set_models_to_train_mode(self):
        """Set models to train mode.

        You can override this method to set your own models to train mode. For example, in GAN training, you may want to
        set the generator and the discriminator to train mode.
        """
        self.model.train()

    def set_models_to_eval_mode(self):
        self.model.eval()

    def lr_scheduler_step(self):
        """Step the lr scheduler.

        You can override this method to step your own lr scheduler. For example, in GAN training, you may want to
        step the lr scheduler of the generator and the discriminator.
        """
        self.lr_scheduler.step()

    def create_bar_desc(self, loss_dict: Dict[str, float]):
        bar_desc = ""
        for k, v in loss_dict.items():
            bar_desc += f"{k}: {(v):.4f}, "
        bar_desc += f"lr: {self.lr_scheduler.get_last_lr()[-1]:.10f}"

        if self.args.plot_lr:
            self.writer.add_scalar(
                "Train_Step/lr",
                self.lr_scheduler.get_last_lr()[-1],
                self.state.steps_trained,
            )

        return bar_desc

    def _has_improved(self, score, save_max_score=True):
        """Check if the current model got the best metric score"""
        if save_max_score:
            return score > self.state.best_score
        else:
            return score < self.state.best_score

    def _should_early_stop(self, score: float):
        should_stop = False

        if self._has_improved(score, save_max_score=self.args.greater_is_better):
            self.state.best_score = score
            self.state.best_score_epoch = self.state.epochs_trained
            self.save_checkpoint(self.state.epochs_trained, is_best_epoch=True)
            self.state.early_stopping_patience_counter = 0
            logger.info(f"Found new best score: {score:.4f}, saving checkpoint...")
        else:
            logger.info(
                f"Score did not improve from {self.state.best_score:.4f} at epoch {self.state.best_score_epoch}."
            )
            self.state.early_stopping_patience_counter += 1
            logger.info(
                f"Early stopping counter: {self.state.early_stopping_patience_counter} out of {self.args.early_stopping_patience}"
            )

            if (
                self.state.early_stopping_patience_counter
                >= self.args.early_stopping_patience
            ):
                logger.info("Early stopping triggered, stopping training...")
                should_stop = True

        return should_stop

    def init_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: `train()` requires a `train_dataset`.")

        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=0,  # This number should be identical across all processes in the distributed group
            )
        else:
            sampler = None

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "sampler": sampler,
            "shuffle": (sampler is None),
        }

        if self.is_rank_zero:
            logger.info("Dataset and Sampler are initialized.")

        return DataLoader(self.train_dataset, **dataloader_params)

    def init_eval_dataloaders(self) -> Dict[str, DataLoader]:
        """Create the evaluation dataloaders.

        If the eval_dataset is a single dataset, it will be converted to a dictionary with the key "default".

        Returns:
            eval_dataloaders: the evaluation dataloaders.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a `eval_dataset`.")

        eval_dataset = self.eval_dataset
        data_collator = self.data_collator

        if not isinstance(eval_dataset, dict):
            if isinstance(eval_dataset, Dataset):
                eval_dataset = {"default": eval_dataset}
            else:
                raise ValueError(
                    "Trainer: `eval_dataset` should be either a dataset or a dictionary of datasets."
                )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": False,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "shuffle": False,  # No need to shuffle for evaluation
        }

        eval_dataloaders = {}
        for key, dataset in eval_dataset.items():
            if dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,  # No need to shuffle for evaluation
                    seed=0,  # This number should be identical across all processes in the distributed group
                )
            else:
                sampler = None

            eval_dataloaders[key] = DataLoader(
                dataset=dataset,
                sampler=sampler,
                **dataloader_params,
            )

        if self.is_rank_zero:
            logger.info("Evaluation dataloaders are initialized.")
            logger.info(f"Number of evaluation dataloaders: {len(eval_dataloaders)}")

        return eval_dataloaders

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArgs,
    ) -> Tuple[torch.optim.Optimizer, Dict]:
        """Returns the optimizer class and optimizer parameters based on the training arguments."""
        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }

        if args.optim == "adamw":
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs |= adam_kwargs
        elif args.optim == "adam":
            optimizer_cls = torch.optim.Adam
            optimizer_kwargs |= adam_kwargs
        else:
            raise ValueError(f"Unknown optimizer: {args.optim}")

        return optimizer_cls, optimizer_kwargs

    def init_optimizer(self):
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

    def get_warmup_steps(self, warmup_steps, max_steps, warmup_ratio):
        """Calculate the number of warmup steps based on the training arguments."""
        if warmup_steps > 0:
            if self.is_rank_zero:
                logger.info(
                    f"warmup_steps={warmup_steps}. warmup_ratio will be ignored."
                )
            return warmup_steps
        else:
            return math.ceil(max_steps * warmup_ratio)

    def create_warmup_scheduler(self, optimizer, scheduler_name, max_steps: int):
        num_warmup_steps = self.get_warmup_steps(
            self.args.warmup_steps, max_steps, self.args.warmup_ratio
        )

        if scheduler_name == "constant_schedule_with_warmup":
            return get_constant_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps
            )
        elif scheduler_name == "linear_schedule_with_warmup":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_steps,
            )
        elif scheduler_name == "cosine_with_min_lr_schedule_with_warmup":
            return get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_steps,
                num_cycles=self.args.cosine_with_min_lr_schedule_with_warmup_num_cycles,
                initial_lr=self.args.cosine_with_min_lr_schedule_with_warmup_initial_lr,
                min_lr=self.args.cosine_with_min_lr_schedule_with_warmup_min_lr,
            )
        else:
            raise ValueError(f"Invalid scheduler name: {scheduler_name}")

    def init_lr_scheduler(
        self, max_steps: int, scheduler_specific_kwargs: Optional[Dict] = None
    ):
        """Setup the learning rate scheduler.

        You can override this method to create your own schedulers. For example, in GAN training, you may want to
        create two schedulers for the generator and the discriminator.

        Args:
            max_steps: the maximum number of steps to train.
        """
        self.lr_scheduler = self.create_warmup_scheduler(
            optimizer=self.optimizer,
            scheduler_name=self.args.lr_scheduler_type,
            max_steps=max_steps,
        )

    def setup_optimizer_and_scheduler(self, max_steps: int):
        """Setup the optimizer and the learning rate scheduler."""
        self.init_optimizer()
        self.init_lr_scheduler(max_steps=max_steps)

    def set_trigger(self):
        """Sets the internal trigger tensor to 1 on the current process. A latter check should follow using this which will check across all processes.

        Notes:
            Does not require `wait_for_everyone()`
        """
        if dist.is_available() and dist.is_initialized():
            self.flag_tensor = torch.tensor(1, device=self.rank)

    def check_trigger(self):
        """Checks if the internal trigger tensor has been set to 1 in any of the processes. If so, will return `True` and
        reset the trigger tensor to 0.

        Notes:
            Requires `wait_for_everyone()`
        """
        if dist.is_available() and dist.is_initialized():
            # Make sure that all processes have set the trigger tensor
            # Now that we are outside `__init__`, we can initialize it if it is `None` on device
            if self.flag_tensor is None:
                self.flag_tensor = torch.tensor(0, device=self.rank)

            dist.all_reduce(self.flag_tensor, op=dist.ReduceOp.SUM)

            if self.flag_tensor.item() >= 1:
                self.flag_tensor = torch.tensor(0, device=self.rank)
                return True

            return False

    def train(self):
        """Main training entry point."""
        self.in_training = True

        cleanup_before_training()

        # Initialize the training dataloader
        train_dataloader = self.init_train_dataloader()

        # Initialize the training variables
        num_update_steps_per_epoch = len(
            train_dataloader
        )  # #samples // batch_size // world_size
        num_update_steps_per_epoch = (
            num_update_steps_per_epoch // self.args.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            # max_steps is set, so we need to calculate the number of epochs based on max_steps
            max_steps = self.args.max_steps
            num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                max_steps % num_update_steps_per_epoch > 0
            )
        else:
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)

        # Load checkpoint before setting up the model
        ckpt_path = self._parse_ckpt_path_or_alias(self.args.resume_from_checkpoint)
        if ckpt_path is not None:
            state_dict = load_state_dict_from_checkpoint(ckpt_path)
            self.model.load_state_dict(state_dict[StateName.MODEL])

        # Prepare the model for DDP
        self.model = self._prepare_ddp_model(self.model)
        # Create optimizer and lr_scheduler
        self.setup_optimizer_and_scheduler(max_steps=max_steps)

        if ckpt_path is not None:
            # Make sure that the optimizer and lr_scheduler are created after the model is wrapped up with DDP
            self.optimizer.load_state_dict(state_dict[StateName.OPTIMIZER])
            self.lr_scheduler.load_state_dict(state_dict[StateName.LR_SCHEDULER])
            self.state.load_state_dict(state_dict[StateName.TRAINER_STATE])

        # Train!!!
        if self.is_rank_zero:
            logger.info("***** Running training *****")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  `steps_per_epoch` = {num_update_steps_per_epoch:,}")
            logger.info(
                f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
            )
            logger.info(
                f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
            )
            logger.info(f"  Total optimization steps = {max_steps:,}")

        for epoch in range(self.state.epochs_trained + 1, num_train_epochs + 1):
            if self.is_rank_zero:
                logger.info(
                    f"{'=' * 9} Epoch {epoch} out of {num_train_epochs} {'=' * 9}"
                )
                logger.info("Begin training...")

            if dist.is_available() and dist.is_initialized():
                train_dataloader.sampler.set_epoch(epoch)

            self.set_models_to_train_mode()

            training_epoch_output = []

            # The iter number of progress bar increments by 1 by default whether gradient accumulation is used or not.
            # but we update the description of the progress bar only when the gradients are synchronized across all processes.
            dataloader_bar = tqdm(
                train_dataloader,
                desc="",
                dynamic_ncols=True,
                bar_format="{l_bar}{r_bar}",
                colour="green",
                disable=not self.is_rank_zero,
                position=0,
                leave=True,
            )

            for batch_idx, batch in enumerate(dataloader_bar):
                # You are responsible for calling `.backward()`, `.step()`, and `.zero_grad()` in your implementation
                loss_dict = self.training_step(batch, batch_idx)
                training_epoch_output.append(loss_dict)
                if self.is_rank_zero:
                    bar_desc = self.create_bar_desc(loss_dict)
                    dataloader_bar.set_description_str(bar_desc)

                self.lr_scheduler_step()
                self.state.steps_trained += 1
            self.state.epochs_trained += 1

            # Clean cache
            # [Not necessary in most cases]
            # Normally, PyTorch will occupy maximum GPU memory after training for a while. This will prevent other
            # processes from other users from using the GPU. It is a safe practice. Tt keeps it in a pool so that next
            # allocations can be done much faster.
            # However, we need to use more memeroy during evaluation (DNSMOS), so we need to clean the cache.
            torch.cuda.empty_cache()

            # Hook `training_epoch_end`
            self.training_epoch_end(training_epoch_output)

            # Should save?
            if self.is_rank_zero and epoch % self.args.save_epoch_interval == 0:
                self.save_checkpoint(epoch=epoch, is_best_epoch=False)

            # Should evaluate?
            if epoch % self.args.eval_epoch_interval == 0:
                if self.is_rank_zero:
                    logger.info("Training finished, begin evaluation...")

                with torch.no_grad():
                    # only the main process will receive the score
                    score = self.evaluate()
                    if self.is_rank_zero:
                        should_stop = self._should_early_stop(score)
                        if should_stop:
                            self.set_trigger()

                if self.is_rank_zero:
                    logger.info("Evaluation finished.")

            wait_for_everyone()
            # Check if any process has set the trigger tensor
            if self.check_trigger():
                break

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation (validation) and returns metrics.

        Returns:
            score: The representative score of the evaluation.
        """
        if self.is_rank_zero:
            logger.info("Begin evaluation...")

        # init_eval_dataloaders() might be called multiple times if the evaluation is called during training.
        # Here, we will make sure that the dataloaders are only prepared once.
        if self.eval_dataloaders is None:
            if self.is_rank_zero:
                logger.info(
                    "The evaluation dataloaders are not initialized. Initialize them now..."
                )
            self.eval_dataloaders = self.init_eval_dataloaders()

        # If the current run is not in training (use train mode), we need to prepare the model for DDP.
        if not self.in_training:
            # Load checkpoint before setting up the model
            ckpt_path = self._parse_ckpt_path_or_alias(self.args.resume_from_checkpoint)

            if ckpt_path is not None:
                if self.is_rank_zero:
                    logger.info(f"Load the checkpoint from {ckpt_path.as_posix()}...")

                state_dict = load_state_dict_from_checkpoint(ckpt_path)
                # Load the model state before wrapping up the model with DDP according to the PyTorch documentation
                self.model.load_state_dict(state_dict[StateName.MODEL])

            # Prepare the model for DDP
            if self.is_rank_zero:
                logger.info("Prepare the model for DDP...")

            self.model = self._prepare_ddp_model(self.model)

        evaluation_output = self.evaluation_loop(
            description="evaluate", gather_step_output=True
        )

        if self.is_rank_zero:
            # only the main process will run evaluation_epoch_end
            logger.info("Evaluation finished, begin hook `evaluation_epoch_end`...")
            score = self.evaluation_epoch_end(evaluation_output)
            return score
        else:
            return None

    @torch.no_grad()
    def predict(self):
        """Run prediction.

        In the predict mode, the model will be loaded from a checkpoint and the evaluation_loop will be called.
        However, the evaluation_loop will not gather any step_output from all processes.
        """
        if self.is_rank_zero:
            logger.info("Begin predicting...")

        # In the predict mode, get_eval_dataloaders() will be called only once.
        self.eval_dataloaders = self.init_eval_dataloaders()

        # Load checkpoint before setting up the model
        ckpt_path = self._parse_ckpt_path_or_alias(self.args.resume_from_checkpoint)
        if ckpt_path is not None:
            state_dict = load_state_dict_from_checkpoint(ckpt_path)

        # Load the model state before wrapping up the model with DDP according to the PyTorch documentation
        self.model.load_state_dict(state_dict[StateName.MODEL])
        self.model = self._prepare_ddp_model(self.model)

        # In the predict mode, we don't need to gather the step_output from all processes.
        self.evaluation_loop(description="predict", gather_step_output=False)

        logger.info("Prediction finished.")

    @torch.no_grad()
    def evaluation_loop(self, description: str, gather_step_output: bool = False):
        """Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`."""
        args = self.args

        self.set_models_to_eval_mode()

        if self.is_rank_zero:
            logger.info(f"***** Running {description} *****")
            logger.info(f"  Batch size = {args.eval_batch_size}")

        evaluation_output = {}
        for dl_idx, (dl_id, dataloader) in enumerate(self.eval_dataloaders.items()):
            dataloader_output = []
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Evaluation on dataloader `{dl_id}`",
                    bar_format="{l_bar}{r_bar}",
                    dynamic_ncols=True,
                    disable=not self.is_rank_zero,
                )
            ):
                """
                It is advised against computing metrics within the `evaluation_epoch_end` method for several reasons:
                    1. Most evaluation metrics are inherently sequential and not parallelizable. Hence, computing them in `evaluation_epoch_end` does not offer a speed advantage during evaluation.
                    2. By not aggregating all outputs for metric calculation at the epoch's end, we reduce the risk of memory overflow, which can occur when gathering results across all processes.
                    3. Calculating the metric score during `evaluation_step` allows for earlier detection of any errors in the code.

                Recommendations for metric calculation:
                    1. Perform immediate metric score calculation within the `evaluation_step` method.
                    2. Accumulate the results at this stage.
                    3. If necessary, compute the average or aggregate metric score in the `evaluation_epoch_end` method.
                """
                with torch.no_grad():
                    step_output = self.evaluation_step(batch, batch_idx, dl_id)

                # If `gather_step_output` is True, we will gather the step_output from all processes and return a list of all metric scores.
                if gather_step_output:
                    """
                    Collect the step_output from all processes and return a list of all metric scores. Assume we have two processes:
                    step_output = [
                        {"metric_1": xx, "metric_2": xx, ...},  # process 0
                        {"metric_1": xx, "metric_2": xx, ...},  # process 1
                        {"metric_1": xx, "metric_2": xx, ...},  # process 0
                        {"metric_1": xx, "metric_2": xx, ...},  # process 1
                        ...
                    ]
                    """
                    step_output = gather_object(step_output, dst=0)
                dataloader_output.append(step_output)
            evaluation_output[dl_id] = dataloader_output

        """
        evaluation_output = {
            "dataloader_id_1": [step_output_0, step_output_1, ...],
            "dataloader_id_2": [step_output_0, step_output_1, ...],
            ...
        }
        """
        return evaluation_output

    def training_step(self, batch, batch_idx):
        """Implement a training step (iteration).

        Implement your own training step here. The input batch is from a training dataloader and the output of this
        function can be various. For example, it can be a dict of loss, or a dict of loss and some enhanced audio signals.
        Here is the persuade code for training a model:

        .. code-block:: python
            :emphasize-lines: 6

            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    loss_dict = training_step(batch, batch_idx)
                    training_epoch_output.append(loss_dict)

                training_epoch_end(training_epoch_output)
                save_checkpoint()
                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)

        Args:
            batch: a batch of data, which passed from a custom training dataloader.
            batch_idx: the index of the current batch.

        Returns:
            loss_dict: a dict of loss. For example, {"loss_1": loss, "loss_2": loss, ...}
        """
        raise NotImplementedError

    def training_epoch_end(self, training_epoch_output):
        """Implement the logic of the end of a training epoch.

        By default, this function will log the mean loss of each loss item on a training epoch. You can override this.
        When the training epoch ends, this function will be called. The input is a list of the loss dict of each step
        in a training epoch. You may want to log the epoch-level training loss here.

        .. code-block:: python
            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    loss = training_step(batch, batch_idx)
                    training_epoch_output.append(loss)

                training_epoch_end(training_epoch_output)
                save_checkpoint()
                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)

        Args:
            training_epoch_output: the output of the training epoch. It may a list of the output of each batch (iteration).
        """
        # Compute mean loss on all loss items on a epoch
        if self.is_rank_zero:
            loss_keys = training_epoch_output[0].keys()
            loss_dict = {
                key: np.mean([step_out[key] for step_out in training_epoch_output])
                for key in loss_keys
            }

            for key, value in loss_dict.items():
                logger.info(
                    f"Loss '{key}' on epoch {self.state.epochs_trained}: {value}"
                )
                self.writer.add_scalar(
                    f"Train_Epoch/{key}", value, self.state.epochs_trained
                )

            # loss_dict["epoch"] = self.state.epochs_trained
            # Append the loss to the loss log
            # df = pd.DataFrame(loss_dict, index=[0])
            # df.to_csv(self.loss_log_path, mode="a", index=False, header=not self.loss_log_path.exists())

    def evaluation_step(self, batch, batch_idx, dataloader_idx):
        """Implement a evaluation/prediction step.

        This function defines the evaluation step. The input batch is from a eval dataloader.
        Here is the persuade code for validating a model:

        .. code-block:: python
            :emphasize-lines: 4

            evaluation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = evaluation_step(batch, batch_idx)
                    evaluation_epoch_output.append(loss_or_data)

            score = evaluation_epoch_end(evaluation_epoch_output)
            return score

        Notes:
            **The evaluation step will be run on all processes.**

            About batch size:
            If your evaluation data have the same length, you may use a batch size larger than 1 to speed up the evaluation.
            For example, if you have 1000 samples in the evaluation set, and you have a batch size of 100, then you will
            have 10 batches in the evaluation set. However, if your data in the evaluation set has a different length, please
            use a batch size of 1. It still works for distributed evaluation. Otherwise, you will get an error.

            About distributed evaluation:
            The output of this function will be gathered across all processes. For example, if you have 4 processes, and
            you have a batch size of 1, then you will have 4 outputs from this function. The output of this function will
            be gathered across all processes. The first dimension of the result is num_processes multiplied by the first
            dimension of the input tensors. **Please make sure the first dimension of the input tensors is the batch size.**
            **The last dimension of the output will be padded to the length of the longest sample in the evaluation set.**
            It means that the output will be a tensor with the shape of [num_processes * batch_size, max_length]. If you
            calculate the metric score on the output, you should do a truncation to remove the padding. Otherwise, if you
            are using a metric that sensitive to the padding, you will get a wrong metric score. It is not easy to
            implement this truncation in the ``evaluation_epoch_end`` function. We recommend you directly calculate the metric
            score in the evaluation_step function. I guess the Accelerate team will implement a automatic truncation in the
            future. https://github.com/huggingface/accelerate/issues/226

        Args:
            batch: a batch of data.
            batch_idx: the index of the batch.
            dataloader_idx: the index of the dataloader.

        Returns:
            output: the output of the batch. It may enhanced audio signals.
        """
        raise NotImplementedError

    def evaluation_epoch_end(self, outputs, log_to_tensorboard=True):
        """evaluation epoch end.

        The input `evaluation_epoch_output` will be a list of list. For example, if you have two dataloaders, the `evaluation_epoch_output` will be:

        .. code-block:: python

            evaluation_epoch_output = [
                [dataloader_1_batch_1_output, dataloader_1_batch_2_output, ...],
                [dataloader_2_batch_1_output, dataloader_2_batch_2_output, ...],
                ...,
            ]


        The output of this function should be a metric score, which will be used to determine whether the current model is the best model.

        .. code-block:: python
            :emphasize-lines: 7

            evaluation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = evaluation_step(batch, batch_idx)
                    evaluation_epoch_output.append(loss_or_data)

            score = evaluation_epoch_end(evaluation_epoch_output)
            return score

        Args:
            evaluation_epoch_output: the output of the evaluation epoch. It is a list of list.

        Returns:
            score: the metric score of the evaluation epoch.
        """
        # We use this variable to store the score for the current epoch
        score = 0.0

        for dl_id, dataloader_outputs in outputs.items():
            metric_dict_list = []
            for i, step_output in enumerate(dataloader_outputs):
                metric_dict_list += step_output
            # Use pandas to compute the mean of all metrics and save them to a csv file
            df_metrics = pd.DataFrame(metric_dict_list)
            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = (
                df_metrics_mean.to_frame().T
            )  # Convert mean to a DataFrame

            time_now = self._current_time_now()
            df_metrics.to_csv(
                self.metrics_dir
                / f"dl_{dl_id}_epoch_{self.state.epochs_trained}_{time_now}.csv",
                index=False,
            )
            df_metrics_mean_df.to_csv(
                self.metrics_dir
                / f"dl_{dl_id}_epoch_{self.state.epochs_trained}_{time_now}_mean.csv",
                index=False,
            )

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            # We use the `metric_for_best_model` to compute the score. In this case, it is the `si_sdr`.
            if self.in_training:
                score += df_metrics_mean[self.args.metric_for_best_model]

                if log_to_tensorboard:
                    for metric, value in df_metrics_mean.items():
                        self.writer.add_scalar(
                            f"metrics_{dl_id}/{metric}",
                            value,
                            self.state.epochs_trained,
                        )

        return score
