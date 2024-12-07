from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple, Union

import safetensors.torch
import torch
import torch.nn as nn
from simple_parsing.helpers import Serializable


@dataclass
class LoraArgs(Serializable):
    rank: int
    scaling: float

    def __post_init__(self):
        assert self.rank > 0
        assert self.scaling > 0.0


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, scaling: float, bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        assert not bias
        self.bias = bias
        self.rank = rank
        self.scaling = scaling

        self.lora_A = nn.Linear(self.in_features, self.rank, bias=self.bias)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=self.bias)

        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        # make sure no LoRA weights are marked as "missing" in load_state_dict
        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            incompatible_keys.missing_keys[:] = []

        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def forward(self, x: torch.Tensor):
        lora = self.lora_B(self.lora_A(x))
        return self.linear(x) + self.scaling * lora

    def _load_from_state_dict(self, state_dict: Dict[str, Any], prefix: str, *args: Any, **kwargs: Any):
        """Reimplement the load_from_state_dict method to handle the LoRA weights.

        `nn.Module.load_state_dict()` calls this method to load the state_dict of the current module.
        We can reimplement this method to map the keys of the state_dict to the correct weights of the module.

        Args:
            state_dict: _description_
            prefix: a string that is used to prefix the keys of the state_dict. Default is an empty string.
        """
        key_name = prefix + "weight"

        # full checkpoint
        # if key_name in state_dict, it means that this is we first time we load this LoRALinear
        # if key_name is not in state_dict, it means that we are resuming from a checkpoint during training LoRALinear
        if key_name in state_dict:
            w_ref = state_dict[key_name]

            # load frozen weights
            state_dict = {
                "linear.weight": w_ref,
                "lora_A.weight": torch.zeros_like(self.lora_A.weight, device=w_ref.device, dtype=w_ref.dtype),
                "lora_B.weight": torch.zeros_like(self.lora_B.weight, device=w_ref.device, dtype=w_ref.dtype),
            }

            self.load_state_dict(state_dict, assign=True, strict=True)


class LoRALoaderMixin:
    def load_lora(self, lora_path: Union[Path, str], scaling: float = 2.0):
        state_dict = safetensors.torch.load_file(lora_path)

        self._load_lora_state_dict(state_dict, scaling=scaling)

    def _load_lora_state_dict(self, lora_state_dict: Dict[str, torch.Tensor], scaling: float = 2.0):
        lora_dtypes = {p.dtype for p in lora_state_dict.values()}
        pass
