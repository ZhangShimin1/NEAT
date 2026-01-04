import sys

sys.path.append("../..")
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dataset import Evaluation_Dataset, Semi_Dataset, Train_Dataset
from simple_parsing import Serializable, parse
from trainer import Trainer

from acouspike.models.model_warpper import ModelWrapper, ModelWrapperArgs
from acouspike.src.accelerate import init_accelerator
from acouspike.src.logger import init_logging_logger
from acouspike.src.trainer_args import TrainingArgs

logger = logging.getLogger(__name__)


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
        train_dataset = Train_Dataset(
            args.data.train_csv_path, args.data.second, aug=args.data.aug, pairs=False
        )
    else:
        train_dataset = Semi_Dataset(
            args.data.train_csv_path,
            args.data.unlabel_csv_path,
            args.data.second,
            aug=args.data.aug,
            pairs=False,
        )
    # Evaluation trials
    trials = np.loadtxt(args.data.trial_path, str)
    eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
    print("number of enroll: {}".format(len(set(trials.T[1]))))
    print("number of test: {}".format(len(set(trials.T[2]))))
    print("number of evaluation: {}".format(len(eval_path)))
    test_dataset = Evaluation_Dataset(eval_path, second=-1)
    in_dim = 80

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
        **args.model.SG_args,
    )
    trials = np.loadtxt(args.data.trial_path, str)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args.trainer,
        train_dataset=train_dataset if args.trainer.do_train else None,
        eval_dataset=test_dataset,
        loss_name=args.loss_name,
        emb_dim=args.emb_dim,
        num_classes=args.num_classes,
        trials=trials,
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
