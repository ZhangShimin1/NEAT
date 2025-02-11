
import torch
import logging
from tqdm import tqdm
import pandas as pd
from acouspike.utils.monitor import OutputMonitor, cal_firing_rate
from acouspike.models.neuron.base_neuron import BaseNeuron
from acouspike.src.accelerate import gather_object
from acouspike.src.trainer import Trainer as BaseTrainer
logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    def __init__(self, *args, cal_FR=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.cal_FR = cal_FR
        if cal_FR:
            self.spike_seq_monitor = OutputMonitor(self.model, BaseNeuron, cal_firing_rate)

    def training_step(self, batch, batch_idx):        
        spect, target = batch
        x = spect.cuda()
        y = target.long().cuda()
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
        spect, target = batch
        x = spect.cuda()
        y = target.long().cuda()
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
                        FR_records[m] = {}
                    FR_temp = torch.cat([self.spike_seq_monitor.records[index] for index in self.spike_seq_monitor.name_records_index[m]], dim=0)
                    FR_records[m]['firing_rate'] = FR_temp.mean().detach().cpu().numpy().item()
                    # Get the module instance from the model using the name
                    module = dict(self.model.module.named_modules())[m]
                    FR_records[m]['module_type'] = module.__class__.__name__
                    FR_records[m]['neuron_num'] = module.neuron_num
                    FR_records[m]['recurrent'] = module.recurrent

                # Convert nested dict to DataFrame format
                df_records = []
                for module_name, stats in FR_records.items():
                    df_records.append({
                        'Neuron index': module_name,
                        'Module Type': stats['module_type'],
                        'Firing Rate': stats['firing_rate'],
                        'Neuron Number': stats['neuron_num'],
                        'Recurrent': stats['recurrent']
                    })
                FR_df = pd.DataFrame(df_records)
                FR_df.to_csv(self.metrics_dir / f"FiringRate_{dl_id}_epoch_{self.state.epochs_trained}.csv", index=False)
        """
        evaluation_output = {
            "dataloader_id_1": [step_output_0, step_output_1, ...],
            "dataloader_id_2": [step_output_0, step_output_1, ...],
            ...
        }
        """
        return evaluation_output
    
