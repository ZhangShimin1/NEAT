import logging
from itertools import chain

import numpy as np
import pandas as pd
import score as metric_score
import torch
from loss import amsoftmax, softmax
from modules.feature import Mel_Spectrogram
from spikingjelly.activation_based.functional import reset_net
from tqdm import tqdm

from neat.models.neuron.base_neuron import BaseNeuron
from neat.src.accelerate import gather_object
from neat.src.trainer import Trainer as BaseTrainer
from neat.utils.monitor import OutputMonitor, cal_firing_rate

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(
        self, loss_name, emb_dim, num_classes, trials, cal_FR=True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if loss_name == "amsoftmax":
            self.loss_fun = amsoftmax(emb_dim, num_classes).cuda()
        else:
            self.loss_fun = softmax(emb_dim, num_classes).cuda()
        self.mel_trans = Mel_Spectrogram()
        self.trials = trials
        self.cal_FR = cal_FR
        if cal_FR:
            self.spike_seq_monitor = OutputMonitor(
                self.model, BaseNeuron, cal_firing_rate
            )

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        feature = self.mel_trans(waveform)
        feature = feature.squeeze(1).permute(0, 2, 1).cuda()
        embedding, states = self.model(feature)
        embedding = embedding.sum(1)
        loss, acc = self.loss_fun(embedding, label.cuda())
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        reset_net(self.model)
        self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "accuracy": acc[0].detach().cpu().numpy(),
        }

    def evaluation_step(self, batch, batch_idx, dl_id):
        x, path = batch
        path = path[0]
        with torch.no_grad():
            feature = self.mel_trans(x)
            feature = feature.squeeze(1).permute(0, 2, 1).cuda()
            self.model.eval()
            embedding, states = self.model(feature)
            embedding = embedding.sum(1)
        reset_net(self.model)
        x = embedding.detach().cpu().numpy()[0]
        return [{"eval_vectors": x, "index_mapping": {path: batch_idx}}]

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
                    FR_temp = torch.cat(
                        [
                            self.spike_seq_monitor.records[index]
                            for index in self.spike_seq_monitor.name_records_index[m]
                        ],
                        dim=0,
                    )
                    FR_records[m]["firing_rate"] = (
                        FR_temp.mean().detach().cpu().numpy().item()
                    )
                    # Get the module instance from the model using the name
                    module = dict(self.model.module.named_modules())[m]
                    FR_records[m]["module_type"] = module.__class__.__name__
                    FR_records[m]["neuron_num"] = module.neuron_num
                    FR_records[m]["recurrent"] = module.recurrent

                # Convert nested dict to DataFrame format
                df_records = []
                for module_name, stats in FR_records.items():
                    df_records.append(
                        {
                            "Neuron index": module_name,
                            "Module Type": stats["module_type"],
                            "Firing Rate": stats["firing_rate"],
                            "Neuron Number": stats["neuron_num"],
                            "Recurrent": stats["recurrent"],
                        }
                    )
                FR_df = pd.DataFrame(df_records)
                FR_df.to_csv(
                    self.metrics_dir
                    / f"FiringRate_{dl_id}_epoch_{self.state.epochs_trained}.csv",
                    index=False,
                )
        """
        evaluation_output = {
            "dataloader_id_1": [step_output_0, step_output_1, ...],
            "dataloader_id_2": [step_output_0, step_output_1, ...],
            ...
        }
        """
        return evaluation_output

    def evaluation_epoch_end(self, outputs, log_to_tensorboard=True):
        score = 0.0

        for dl_id, dataloader_outputs in outputs.items():
            metric_dict_list = []
            for i, step_output in enumerate(dataloader_outputs):
                metric_dict_list += step_output

            eval_vectors = np.vstack(
                [item["eval_vectors"] for item in metric_dict_list]
            )
            index_mapping = dict(
                chain.from_iterable(
                    item["index_mapping"].items() for item in metric_dict_list
                )
            )
            eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
            labels, scores = metric_score.cosine_score(
                self.trials, index_mapping, eval_vectors
            )

            EER, threshold = metric_score.compute_eer(labels, scores)
            print(
                "\ncosine EER: {:.2f}% with threshold {:.2f}".format(
                    EER * 100, threshold
                )
            )
            logger.info("cosine_eer: %.2f%%", EER * 100)
            df_metrics = pd.DataFrame(
                [{"cosine_eer": float(EER * 100), "threshold": float(threshold.item())}]
            )

            minDCF, threshold = metric_score.compute_minDCF(
                labels, scores, p_target=0.01
            )
            print(
                "cosine minDCF(10-2): {:.2f} with threshold {:.2f}".format(
                    minDCF, threshold
                )
            )
            logger.info("cosine_minDCF(10-2): %.4f", minDCF)

            minDCF, threshold = metric_score.compute_minDCF(
                labels, scores, p_target=0.001
            )
            print(
                "cosine minDCF(10-3): {:.2f} with threshold {:.2f}".format(
                    minDCF, threshold
                )
            )
            logger.info("cosine_minDCF(10-3): %.4f", minDCF)

            time_now = self._current_time_now()
            df_metrics.to_csv(
                self.metrics_dir
                / f"dl_{dl_id}_epoch_{self.state.epochs_trained}_{time_now}.csv",
                index=False,
            )
            logger.info(f"\n{df_metrics.to_markdown()}")

            # We use the `metric_for_best_model` to compute the score. In this case, it is the `si_sdr`.
            if self.in_training:
                score += df_metrics[self.args.metric_for_best_model].iloc[0]

                if log_to_tensorboard:
                    for metric, value in df_metrics.items():
                        self.writer.add_scalar(
                            f"metrics_{dl_id}/{metric}",
                            value.iloc[0],
                            self.state.epochs_trained,
                        )
        return score
