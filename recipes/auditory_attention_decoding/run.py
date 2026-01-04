import sys

sys.path.append("../..")
import logging
from dataclasses import dataclass
from pathlib import Path

from dataloader import AADWSDataset
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
    dataset: str
    mode: str
    window: str


@dataclass
class Args(Serializable):
    data: DataArgs
    trainer: TrainingArgs
    model: ModelWrapperArgs


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
            args.trainer.output_dir = (
                f"{root_save_dir}/{args.data.dataset}/subj-{sub}/fold-{i}"
            )
            init_logging_logger(args.trainer.output_dir)
            args.save(Path(args.trainer.output_dir) / "conf.yaml")
            aad = AADWSDataset(dataset=args.data.dataset, subject_id=sub, fold_num=i)
            train_dataset, val_dataset, test_dataset = aad.get_datasets()
            # Initialize model
            in_dim, out_dim = 64, 2
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
                **args.model.SG_args,
            )
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=args.trainer,
                train_dataset=train_dataset if args.trainer.do_train else None,
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
