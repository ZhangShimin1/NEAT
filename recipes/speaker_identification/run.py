import sys

sys.path.append("../..")
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from raw_transform import test_transforms, train_transforms
from simple_parsing import Serializable, parse
from trainer import Trainer
from vox1_dataset import RawWaveformDataset

from acouspike.models.model_warpper import ModelWrapper, ModelWrapperArgs
from acouspike.src.accelerate import init_accelerator
from acouspike.src.logger import init_logging_logger
from acouspike.src.trainer_args import TrainingArgs

logger = logging.getLogger(__name__)


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

    audio_config = {
        "feature": "raw",
        "normalize": False,
        "sample_rate": 16000,
        "min_duration": 1,
        "random_clip_size": 1,
        "val_clip_size": 1,
        "mixup": False,
    }
    labels_delimiter = ","
    mode = "multiclass"
    sample_rate = 16000
    random_clip_size = 16000
    val_clip_size = 16000
    meta_dir = "voxceleb1_meta"
    tr_tfs = train_transforms(True, random_clip_size, sample_rate=sample_rate)
    val_tfs = train_transforms(False, val_clip_size, sample_rate=sample_rate)
    test_tfs = test_transforms(sample_rate=sample_rate)

    train_dataset = RawWaveformDataset(
        os.path.join(meta_dir, "train.csv"),
        os.path.join(meta_dir, "lbl_map.json"),
        audio_config,
        mode=mode,
        augment=True,
        mixer=None,
        delimiter=labels_delimiter,
        transform=tr_tfs,
        is_val=False,
        cropped_read=False,
    )

    # val_set = RawWaveformDataset(os.path.join(meta_dir, "val.csv"),
    #                             os.path.join(meta_dir, "lbl_map.json"),
    #                                 audio_config,
    #                                 mode=mode, augment=False,
    #                                 mixer=None, delimiter=labels_delimiter,
    #                                 transform=val_tfs, is_val=True)

    test_dataset = RawWaveformDataset(
        os.path.join(meta_dir, "test.csv"),
        os.path.join(meta_dir, "lbl_map.json"),
        audio_config,
        mode=mode,
        transform=test_tfs,
        is_val=True,
        delimiter=labels_delimiter,
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
