from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
sys.path.append('/home/zysong/AcouSpike')
sys.path.append('/home/zysong/AcouSpike/src')
import logging
import torch
import torch.distributed as dist
import os
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from src.audiozen.accelerate import init_accelerator
import importlib
from src.audiozen.logger import init_logging_logger
from src.audiozen.trainer import Trainer as BaseTrainer
from src.audiozen.trainer_args import TrainingArgs
from acouspike.models.model_warpper import ModelWrapper, ModelWrapperArgs
from simple_parsing import Serializable, parse
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from dataset import Evaluation_Dataset, Train_Dataset, Semi_Dataset
from spikingjelly.activation_based.functional import reset_net
import score as metric_score
import pandas as pd
from loss import softmax, amsoftmax
from modules.feature import Mel_Spectrogram
from itertools import chain
logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, loss_name, emb_dim, num_classes, trials, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_name == "amsoftmax":
            self.loss_fun = amsoftmax(emb_dim, num_classes).cuda()
        else:
            self.loss_fun = softmax(emb_dim, num_classes).cuda()
        self.mel_trans = Mel_Spectrogram()
        self.trials = trials

    
    def training_step(self, batch, batch_idx): 
        waveform, label = batch
        feature = self.mel_trans(waveform)
        feature = feature.squeeze(1).permute(0, 2, 1).cuda()
        embedding, states = self.model(feature)
        embedding = embedding.mean(1)
        loss, acc = self.loss_fun(embedding, label.cuda())
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        reset_net(self.model)
        self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "accuracy": acc[0].detach().cpu().numpy()
        }

    def evaluation_step(self, batch, batch_idx, dl_id):
        x, path = batch
        path = path[0]
        with torch.no_grad():
            feature = self.mel_trans(x)
            feature = feature.squeeze(1).permute(0, 2, 1).cuda()
            self.model.eval()
            embedding, states = self.model(feature)
            embedding = embedding.mean(1)
        reset_net(self.model)
        x = embedding.detach().cpu().numpy()[0]
        return[{
            "eval_vectors": x,
            "index_mapping": {path: batch_idx}
        }]
    
    def evaluation_epoch_end(self, outputs, log_to_tensorboard=True):
        score = 0.0

        for dl_id, dataloader_outputs in outputs.items():
            metric_dict_list = []
            for i, step_output in enumerate(dataloader_outputs):
                metric_dict_list += step_output


            eval_vectors = np.vstack([item["eval_vectors"] for item in metric_dict_list])
            index_mapping = dict(chain.from_iterable(item["index_mapping"].items() for item in metric_dict_list))
            eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
            labels, scores = metric_score.cosine_score(
                self.trials, index_mapping, eval_vectors)
            
            EER, threshold = metric_score.compute_eer(labels, scores)
            print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(EER*100, threshold))
            logger.info("cosine_eer: %.2f%%", EER * 100) 
            df_metrics = pd.DataFrame([{"cosine_eer":float(EER * 100),  "threshold": float(threshold.item())}])

            minDCF, threshold = metric_score.compute_minDCF(labels, scores, p_target=0.01)
            print("cosine minDCF(10-2): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
            logger.info("cosine_minDCF(10-2): %.4f", minDCF)

            minDCF, threshold = metric_score.compute_minDCF(labels, scores, p_target=0.001)
            print("cosine minDCF(10-3): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
            logger.info("cosine_minDCF(10-3): %.4f", minDCF)

            
            time_now = self._current_time_now()
            df_metrics.to_csv(
                self.metrics_dir / f"dl_{dl_id}_epoch_{self.state.epochs_trained}_{time_now}.csv", index=False
            )
            logger.info(f"\n{df_metrics.to_markdown()}")

            # We use the `metric_for_best_model` to compute the score. In this case, it is the `si_sdr`.
            if self.in_training:
                score += df_metrics[self.args.metric_for_best_model].iloc[0]

                if log_to_tensorboard:
                    for metric, value in df_metrics.items():
                        self.writer.add_scalar(f"metrics_{dl_id}/{metric}", value.iloc[0], self.state.epochs_trained)
        return score

# ==================== Entry ====================
@dataclass
class DataArgs(Serializable):
    train_csv_path: str = "metadata/train_vox1.csv"
    trial_path: str = "metadata/vox1_test.txt"
    unlabel_csv_path: str = None
    aug: bool = False
    second: int = 3

@dataclass
class Args(Serializable):
    trainer: TrainingArgs
    model: ModelWrapperArgs
    data: DataArgs
    loss_name: str
    emb_dim: int
    num_classes: int 

def run(args: Args):
    init_accelerator(device=args.trainer.device)

    # Initialize logger
    init_logging_logger(args.trainer.output_dir)

    # Serialize arguments and save to a yaml file
    args.save(Path(args.trainer.output_dir) / "conf.yaml")

    if args.data.unlabel_csv_path is None:
            train_dataset = Train_Dataset(args.data.train_csv_path, 
                                          args.data.second,
                                          aug = args.data.aug,
                                          pairs=False
                                          )
    else:
        train_dataset = Semi_Dataset(args.data.train_csv_path,
                                     args.data.unlabel_csv_path, 
                                     args.data.second, 
                                     aug =args.data.aug,
                                     pairs=False
                                     )
    #Evaluation trials
    trials = np.loadtxt(args.data.trial_path, str)
    eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
    print("number of enroll: {}".format(len(set(trials.T[1]))))
    print("number of test: {}".format(len(set(trials.T[2]))))
    print("number of evaluation: {}".format(len(eval_path)))
    test_dataset = Evaluation_Dataset(eval_path, second=-1)
    in_dim=80

    # Initialize model
    model = ModelWrapper(
                model_name=args.model.model_name,
                input_size=in_dim,
                hidden_size=args.model.hidden_size,
                output_size=args.emb_dim,
                num_layers=args.model.num_layers,
                dropout=args.model.dropout,
                bn=args.model.bn,
                neuron_type=args.model.neuron_type,
                bidirectional=args.model.bidirectional,
                batch_first=args.model.batch_first,
                **args.model.neuron_args,
                **args.model.SG_args
            )
    trials = np.loadtxt(args.data.trial_path, str)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args.trainer,
        train_dataset=train_dataset
        if args.trainer.do_train
        else None,
        eval_dataset=test_dataset,
        loss_name=args.loss_name,
        emb_dim=args.emb_dim,
        num_classes=args.num_classes,
        trials =trials
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
