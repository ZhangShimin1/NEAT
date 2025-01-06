import yaml
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import List

from spikingjelly.activation_based import layer, neuron, surrogate
from .neuron import *
from .sg import SurrogateGradient as SG

from simple_parsing import Serializable

@dataclass
class NetworkArgs(Serializable):
    num_hidden_layers: int
    num_hidden_units: List[int]
    recurrent_layer: List[int]
@dataclass
class NeuronArgs(Serializable):
    neuron_class: str  # Name of the neuron class as a string
    args: dict  # A dictionary to store neuron-specific arguments
    
@dataclass
class ModelArgs(Serializable):
    network: NetworkArgs
    neuron: NeuronArgs

class BaseNet(nn.Module):
    def __init__(self, in_dim, out_dim, T, model_config):
        super().__init__()

        self.network_config = model_config.network
        self.num_hidden_units = self.network_config.num_hidden_units
        self.num_hidden_layers = self.network_config.num_hidden_layers
        self.recurrent_layer = self.network_config.recurrent_layer

        self.neuron_config = model_config.neuron
        
        #[TODO] Consider use "from src.audiozen.utils import instantiate" for safe implementation
        self.neuron_class = globals()[self.neuron_config.neuron_class]  # Dynamically get the neuron class
        self.neuron_args = self.neuron_config.args.copy()  # Neuron-specific arguments
        # Initiate the surrogate gradient function accoriding to the name
        if self.neuron_config.args['surro_grad']:
            self.neuron_args.update({'surro_grad': SG(func_name=self.neuron_config.args['surro_grad'])})  # Add shared arguments if needed

        layers = []
        # Input layer
        input_layer = layer.Linear(in_dim, self.num_hidden_units[0], bias=False)
        layers.append(input_layer)
        layers.append(self.neuron_class(neuron_num=self.num_hidden_units[0], recurrent=False, **self.neuron_args))

        # Hidden layers
        for i in range(1, self.num_hidden_layers):
            in_features = self.num_hidden_units[i - 1]
            out_features = self.num_hidden_units[i]
            linear_layer = layer.Linear(in_features, out_features, bias=False)
            layers.append(linear_layer)
            if i in self.recurrent_layer:
                layers.append(self.neuron_class(neuron_num=out_features, recurrent=True, **self.neuron_args))
            else:
                layers.append(self.neuron_class(neuron_num=out_features, recurrent=False, **self.neuron_args))

        # Output layer
        output_layer = layer.Linear(self.num_hidden_units[-1], out_dim, bias=False)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):  # T, B, F
        output = self.network(x)
        return output.sum(0)
