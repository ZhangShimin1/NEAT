

import torch
import torch.distributed as dist
import logging
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
from acouspike.src.trainer import Trainer as BaseTrainer
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from acouspike.utils.monitor import OutputMonitor, cal_firing_rate
from acouspike.src.accelerate import gather_object
from acouspike.models.neuron.base_neuron import BaseNeuron

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
    def __init__(self, *args, cal_FR=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.cal_FR = cal_FR
        if cal_FR:
            self.spike_seq_monitor = OutputMonitor(self.model, BaseNeuron, cal_firing_rate)
    
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
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == y).float().mean()
        return [{
            "loss": loss.detach().cpu().numpy().item(),
            "test_accuracy": accuracy.detach().cpu().numpy().item(),
        }]
    
    @torch.no_grad()
    def evaluation_loop(self, description: str, gather_step_output: bool = False):
        """Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`."""
        args = self.args

        self.set_models_to_eval_mode()

        if self.is_rank_zero:
            logger.info(f"***** Running {description} *****")
            logger.info(f"  Batch size = {args.eval_batch_size}")

        evaluation_output = {}
        if self.cal_FR:
            FR_records = {}
            FR_output = {}
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
                self.spike_seq_monitor.clear_recorded_data()
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
            if self.cal_FR:
                for m, i in self.spike_seq_monitor.name_records_index.items():
                    if m not in FR_records.keys():
                        FR_records[m] = []
                    FR_temp = torch.cat([self.spike_seq_monitor.records[index] for index in self.spike_seq_monitor.name_records_index[m]], dim=0)
                    FR_records[m] = FR_temp.mean().detach().cpu().numpy().item()
            FR_df = pd.DataFrame(list(FR_records.items()), columns=['Neuron index', 'Firing Rate'])
            FR_df.to_csv(self.metrics_dir / f"FiringRate_{dl_id}_epoch_{self.state.epochs_trained}.csv", index=False)
        """
        evaluation_output = {
            "dataloader_id_1": [step_output_0, step_output_1, ...],
            "dataloader_id_2": [step_output_0, step_output_1, ...],
            ...
        }
        """
        return evaluation_output
    
