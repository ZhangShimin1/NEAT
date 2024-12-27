from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
sys.path.append('../..')
sys.path.append('../../src')

import torch

from src.audiozen.accelerate import init_accelerator
from src.audiozen.logger import init_logging_logger
from src.audiozen.trainer import Trainer as BaseTrainer
from src.audiozen.trainer_args import TrainingArgs
from src.audiozen.utils import *

from acouspike.models.network import *
from simple_parsing import Serializable, parse
from dataloader import AADWSDataset

from spikingjelly.activation_based.functional import reset_net

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        spect, target = batch
        x = spect.squeeze(1).permute(0, 2, 1).cuda()
        y = target.cuda()
        # forward
        logits = self.model(x)
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
    
    def evaluation_step(self, batch, batch_idx, dataloader_idx):
        spect, target = batch
        x = spect.squeeze(1).permute(0, 2, 1).cuda()
        y = target.cuda()
        # forward
        logits = self.model(x)
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
    mode: str
    window: str

@dataclass
class Args(Serializable):
    data: DataArgs
    trainer: TrainingArgs
    model: ModelArgs

def run(args: Args):
    init_accelerator(device=args.trainer.device)
    num_subject = 16 if args.data.dataset == "KUL" else 18
    root_save_dir = args.trainer.output_dir
    for sub in range(1, num_subject + 1):
        sub = str(sub)
        for i in range(1, 6):  # 5-fold cross validation
            # TODO: optimize logger
            print(f"Experiment for subject-{sub}, fold-{i}")
            # Initialize logger
            args.trainer.output_dir = f"{root_save_dir}/{args.data.dataset}/subj-{sub}/fold-{i}"
            init_logging_logger(args.trainer.output_dir)
            # Serialize arguments and save to a yaml file
            args.save(Path(args.trainer.output_dir) / "conf.yaml")
            # Initialize datasets for each sub/fold
            aad = AADWSDataset(dataset=args.data.dataset, subject_id=sub, fold_num=i)
            train_dataset, val_dataset, test_dataset = aad.get_datasets()
            # Initialize model
            in_dim, out_dim = 64, 2
            model = LSTMNet(in_dim, out_dim, args.model)
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=args.trainer,
                train_dataset=train_dataset if args.trainer.do_train else None,
                eval_dataset=test_dataset
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