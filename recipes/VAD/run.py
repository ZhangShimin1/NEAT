from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
sys.path.append('../..')
sys.path.append('../../src')
import os
import json
import torch
from src.audiozen.accelerate import init_accelerator
import importlib
from src.audiozen.logger import init_logging_logger
from src.audiozen.trainer import Trainer as BaseTrainer
from src.audiozen.trainer_args import TrainingArgs
from acouspike.models.network import BaseNet_cupy, ModelArgs
from simple_parsing import Serializable, parse

from spikingjelly.activation_based.functional import reset_net

from dataset import vadDataset

def calculateHTER(y_pred, y):
    '''
     half-total error rate: HTER = 1/2(MR + FAR)
     miss rate: MR = no. of speech sample not detected / total number of speech samples
     false alarm rate: FAR = no. of nonspeech samples detected as speech/ totoal number of non speech samples
    '''
    total_sample = y.size(0)
    total_speech_samples = y.sum().item()
    non_speech = 1-y
    total_nonspeech_samples = total_sample - total_speech_samples
    mr = (total_speech_samples - y_pred.masked_select(y.bool()).sum().item()) / total_speech_samples
    far = y_pred.masked_select(non_speech.bool()).sum().item() / total_nonspeech_samples

    return mr, far, (mr+far)/2

def load_json(path):
    """
    Load json as python object
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):   
        inputs, labels = batch[0].type(torch.FloatTensor).cuda(), \
                         batch[1].type(torch.LongTensor).cuda()
        # forward
        labels = labels.flatten(0, 1)
        logits = self.model(inputs.permute(1, 0, 2)).permute(1, 0, 2).flatten(0,1)
        loss = self.loss_function(logits, labels)
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        reset_net(self.model)
        self.optimizer.step()
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        mr, far, hter = calculateHTER(preds, labels)
        return {
            "loss": loss.detach().cpu().numpy(),
            "accuracy": accuracy.detach().cpu().numpy(),
            "mr": mr,
            "far": far,
            "hter": hter,
        }

    def evaluation_step(self, batch, batch_idx, dl_id):
        inputs, labels = batch[0].type(torch.FloatTensor).cuda(), \
                         batch[1].type(torch.LongTensor).cuda()
        # forward
        labels = labels.flatten(0, 1)
        logits = self.model(inputs.permute(1, 0, 2)).permute(1, 0, 2).flatten(0,1)
        loss = self.loss_function(logits, labels)
        reset_net(self.model)
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        mr, far, hter = calculateHTER(preds, labels)
        return [{
            "loss": loss.detach().cpu().numpy().item(),
            "test_accuracy": accuracy.detach().cpu().numpy().item(),
            "mr": mr,
            "far": far,
            "hter": hter,
        }]
    
# ==================== Entry ====================
@dataclass
class DataArgs(Serializable):
    data_dir: str

@dataclass
class Args(Serializable):
    trainer: TrainingArgs
    model: ModelArgs
    data: DataArgs


def run(args: Args):
    data_path_object = load_json(args.data.data_dir)
    init_accelerator(device=args.trainer.device)

    # Initialize logger
    init_logging_logger(args.trainer.output_dir)

    # Serialize arguments and save to a yaml file
    args.save(Path(args.trainer.output_dir) / "conf.yaml")

    train_dataset = vadDataset(data_path_object[2]['train'], is1dconv=False, isMask=False)
    test_dataset = vadDataset(data_path_object[2]['test'], is1dconv=False, isMask=False)
    in_dim = 40
    out_dim = 2
    # Initialize model
    model = BaseNet_cupy(in_dim, out_dim, args.model)

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
