import math
from functools import partial
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    initial_lr: float = 0.0,
    min_lr: Optional[float] = None,
) -> LambdaLR:
    """
    Create a learning rate scheduler that linearly increases learning rate from 0 to `lr` over `num_warmup_steps`, then
    decreases learning rate from `lr` to 0.0 on a cosine schedule over the remaining `num_training_steps - num_warmup_steps`
    steps (assuming in num_cycles=0.5).

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for which to linearly increase the learning rate.
        num_training_steps: The total number of training steps.
        num_cycles: _description_. Defaults to 0.5.
        last_epoch: _description_. Defaults to -1.

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """
    min_lr_rate = min_lr / initial_lr

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        # move cosine curve to the top and scale it down from 0 to 1
        cosine_lr_multiple = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

        # scale down to min_lr_rate
        cosine_lr_multiple = cosine_lr_multiple * (1.0 - min_lr_rate) + min_lr_rate

        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
