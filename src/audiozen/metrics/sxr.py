from typing import Union

import numpy as np
import torch

from audiozen.metrics.metric_utils import Metric


class SISDR:
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        estimate: Union[torch.tensor, np.ndarray],
        target: Union[torch.tensor, np.ndarray],
        reduce_mean: bool = True,
    ) -> dict:
        if isinstance(estimate, np.ndarray):
            estimate = torch.tensor(estimate)

        if isinstance(target, np.ndarray):
            target = torch.tensor(target)

        eps = torch.finfo(estimate.dtype).eps

        # zero mean to ensure scale invariance
        s_target = target - torch.mean(target, dim=-1, keepdim=True)
        s_estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

        # <s, s'> / ||s||**2 * s
        pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
        s_target_norm = torch.sum(s_target**2, dim=-1, keepdim=True)
        pair_wise_proj = (pair_wise_dot * s_target + eps) / (s_target_norm + eps)

        e_noise = s_estimate - pair_wise_proj

        pair_wise_sdr = (torch.sum(pair_wise_proj**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps)
        val = 10 * torch.log10(pair_wise_sdr + eps)

        if reduce_mean:
            val = torch.mean(val)

        val = val.detach().item()

        return Metric(name="si_sdr", value=val)
