from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
sys.path.append('../..')
sys.path.append('../../src')
sys.path.insert(0, "/home/zysong/AcouSpike/acouspike")
import torch
from acouspike.src.accelerate import init_accelerator
import importlib
from acouspike.src.logger import init_logging_logger
from acouspike.src.trainer import Trainer as BaseTrainer
from acouspike.src.trainer_args import TrainingArgs
from acouspike.models.model_warpper import ModelWrapper, ModelWrapperArgs
from simple_parsing import Serializable, parse

from spikingjelly.activation_based.functional import reset_net

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):        
        spect, target = batch
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
        spect, target = batch
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
class DataArgs(Serializable):
    dataset: str
    aug: bool
    version: int
    if_command: bool
    if_spk: bool

@dataclass
class Args(Serializable):
    trainer: TrainingArgs
    model: ModelWrapperArgs
    data: DataArgs


def run(args: Args):
    init_accelerator(device=args.trainer.device)

    # Initialize logger
    init_logging_logger(args.trainer.output_dir)

    # Serialize arguments and save to a yaml file
    args.save(Path(args.trainer.output_dir) / "conf.yaml")

    # Initialize datasets
    module = importlib.import_module(f'datasets.{args.data.dataset}')
    if args.data.dataset == "gsc":
        gsc_class = getattr(module, "Datasets")
        train_dataset = gsc_class(split="train", version=args.data.version, if_command=args.data.if_command, aug=args.data.aug)
        valid_dataset = gsc_class(split="valid", version=args.data.version, if_command=args.data.if_command, aug=False)
        test_dataset = gsc_class(split="test", version=args.data.version, if_command=args.data.if_command, aug=False)
        in_dim = 40
        T = 98
        if args.data.version == 1:
            if args.data.if_command:
                out_dim = 10
            else:
                out_dim = 30
        else:
            if args.data.if_command:
                out_dim = 14
            else:
                out_dim = 35
    elif args.data.dataset == "shd":
        if args.data.if_spk:
            T = 100
            shd_class = getattr(module, "SpikingDatasets")
            train_dataset = shd_class(split="train")
            test_dataset = shd_class(split="test")
            in_dim = 700
        else:
            hd_class = getattr(module, "NonSpikingDatasets")
            train_dataset = hd_class(split="train", aug=args.data.aug)
            test_dataset = hd_class(split="test", aug=False)
            in_dim = 40
            # TODO: The timestep of each batch is different, which will result in conflict when instantiating the network
            raise ValueError
        out_dim = 20
    elif args.data.dataset == "ssc":
        T = 100
        ssc_class = getattr(module, "SpikingDatasets")
        train_dataset = ssc_class(split="train")
        valid_dataset = ssc_class(split="valid")
        test_dataset = ssc_class(split="test")
        in_dim = 700
        out_dim = 35


    # Initialize model
    model = ModelWrapper(
                model_name=args.model.model_name,
                input_size=in_dim,
                hidden_size=args.model.hidden_size,
                output_size=out_dim,
                num_layers=args.model.num_layers,
                bn=args.model.bn,
                dropout=args.model.dropout,
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
