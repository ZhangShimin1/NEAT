from dataclasses import dataclass
from functools import partial
from pathlib import Path
import logging
import torch
import torch.distributed as dist
import os
from typing import Any, Dict, Optional, Tuple, Union
from acouspike.src.accelerate import init_accelerator
import importlib
from acouspike.src.logger import init_logging_logger
from acouspike.src.trainer import Trainer as BaseTrainer
from acouspike.src.trainer_args import TrainingArgs
from acouspike.models.model_warpper import ModelWrapper, ModelWrapperArgs
from simple_parsing import Serializable, parse
from raw_transform import train_transforms, test_transforms
from vox1_dataset import RawWaveformDataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from spikingjelly.activation_based.functional import reset_net

logger = logging.getLogger(__name__)


def _collate_fn_raw_multiclass(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, channel_size, max_seqlength)
    feats = []
    targets = torch.LongTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        feats.append(sample[1])
        target = sample[2]
        seq_length = real_tensor.size(1)
        # inputs[x] = real_tensor
        inputs[x].narrow(1, 0, seq_length).copy_(real_tensor)
        targets[x] = target
    input_feature = torch.cat(feats, dim=0)
    return inputs, input_feature, targets


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()
    
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
            "collate_fn": _collate_fn_raw_multiclass
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
                raise ValueError("Trainer: `eval_dataset` should be either a dataset or a dictionary of datasets.")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": False,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "shuffle": False,  # No need to shuffle for evaluation
            "collate_fn": _collate_fn_raw_multiclass
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

    def training_step(self, batch, batch_idx):        
        raw_wav, spect, target = batch
        x = spect.cuda()
        y = target.cuda()
        # forward
        logits, states = self.model(x)
        logits = logits.mean(dim=1)
        loss = self.loss_function(logits, y)
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        reset_net(self.model)
        self.optimizer.step()
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == y).float().mean()

        return {
            "loss": loss.detach().cpu().numpy(),
            "accuracy": accuracy.detach().cpu().numpy(),
        }

    def evaluation_step(self, batch, batch_idx, dl_id):
        raw_wav, spect, target = batch
        x = spect.cuda()
        y = target.cuda()
        # forward
        logits, states = self.model(x)
        logits = logits.mean(dim=1)
        loss = self.loss_function(logits, y)
        reset_net(self.model)
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == y).float().mean()
        return [{
            "loss": loss.detach().cpu().numpy().item(),
            "test_accuracy": accuracy.detach().cpu().numpy().item(),
        }]

# ==================== Entry ====================
@dataclass
class Args(Serializable):
    trainer: TrainingArgs
    model: ModelWrapperArgs

def run(args: Args):
    init_accelerator(device=args.trainer.device)

    # Initialize logger
    init_logging_logger(args.trainer.output_dir)

    # Serialize arguments and save to a yaml file
    args.save(Path(args.trainer.output_dir) / "conf.yaml")

    audio_config = {'feature': 'raw', 'normalize': False, 'sample_rate': 16000, 'min_duration': 1, 'random_clip_size': 1, 'val_clip_size': 1, 'mixup': False}
    labels_delimiter = ','
    mode = 'multiclass'
    sample_rate = 16000
    random_clip_size = 16000
    val_clip_size = 16000
    meta_dir = 'voxceleb1_meta'
    tr_tfs = train_transforms(True, random_clip_size,
                                            sample_rate=sample_rate)
    val_tfs = train_transforms(False, val_clip_size,
                                            sample_rate=sample_rate)
    test_tfs = test_transforms(sample_rate=sample_rate)

    train_dataset = RawWaveformDataset(os.path.join(meta_dir, "train.csv"),
                                    os.path.join(meta_dir, "lbl_map.json"),
                                    audio_config,
                                    mode=mode, augment=True,
                                    mixer=None, delimiter=labels_delimiter,
                                    transform=tr_tfs, is_val=False, cropped_read=False)

    # val_set = RawWaveformDataset(os.path.join(meta_dir, "val.csv"),
    #                             os.path.join(meta_dir, "lbl_map.json"),
    #                                 audio_config,
    #                                 mode=mode, augment=False,
    #                                 mixer=None, delimiter=labels_delimiter,
    #                                 transform=val_tfs, is_val=True)
    
    test_dataset = RawWaveformDataset(os.path.join(meta_dir, "test.csv"),
                                os.path.join(meta_dir, "lbl_map.json"),
                                audio_config,
                            mode=mode,
                                transform=test_tfs, is_val=True, delimiter=labels_delimiter
                                )
    
    in_dim = 40
    out_dim = 1251
    # Initialize model
    model = ModelWrapper(
                model_name=args.model.model_name,
                input_size=in_dim,
                hidden_size=args.model.hidden_size,
                output_size=out_dim,
                num_layers=args.model.num_layers,
                dropout=args.model.dropout,
                bn=args.model.bn,
                neuron_type=args.model.neuron_type,
                bidirectional=args.model.bidirectional,
                batch_first=args.model.batch_first,
                **args.model.neuron_args,
                **args.model.SG_args
            )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args.trainer,
        train_dataset=train_dataset
        if args.trainer.do_train
        else None,
        eval_dataset=test_dataset,
    )

    if args.trainer.do_eval:
        trainer.evaluate()
    elif args.trainer.do_predict:
        trainer.predict()
    elif args.trainer.do_train:
        trainer.train()
    else:
        raise ValueError(
            "At least one of `do_train`, `do_eval`, or `do_predict` must be True."
        )


if __name__ == "__main__":
    args = parse(Args, add_config_path_arg=True)
    run(args)
