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
    rest: float
    decay: float
    threshold: float
    surro_grad: str
    exec_mode: str
    
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
        self.general_neuron_args = {
            'rest': self.neuron_config.rest,
            'decay': self.neuron_config.decay,
            'threshold': self.neuron_config.threshold,
            'exec_mode': self.neuron_config.exec_mode,
            'time_step': T,
            'surro_grad': SG(func_name=self.neuron_config.surro_grad)  
            # TODO: each function of sg has empty hyper-para, only triangle is set currently for testing
        }

        layers = []
        # Input layer
        input_layer = layer.Linear(in_dim, self.num_hidden_units[0], bias=False)
        layers.append(input_layer)
        layers.append(Recurrent_LIF(neuron_num=self.num_hidden_units[0], recurrent=False, **self.general_neuron_args))

        # Hidden layers
        for i in range(1, self.num_hidden_layers):
            in_features = self.num_hidden_units[i - 1]
            out_features = self.num_hidden_units[i]
            linear_layer = layer.Linear(in_features, out_features, bias=False)
            layers.append(linear_layer)
            if i in self.recurrent_layer:
                layers.append(Recurrent_LIF(neuron_num=out_features, recurrent=True, **self.general_neuron_args))
            else:
                layers.append(Recurrent_LIF(neuron_num=out_features, recurrent=False, **self.general_neuron_args))
                
        # Output layer
        output_layer = layer.Linear(self.num_hidden_units[-1], out_dim, bias=False)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):  # T, B, F
        output = self.network(x)

        return output.sum(0)



# Basic SNN network based on SpikingJelly
class SJBaseNet(nn.Module):
    def __init__(self, in_dim, out_dim, model_config):
        super().__init__()
        self.network_config = model_config.network
        self.neuron_config = model_config.neuron

        self.num_hidden_units = self.network_config.num_hidden_units
        self.num_hidden_layers = self.network_config.num_hidden_layers
        self.recurrent_layer = self.network_config.recurrent_layer

        self.tau = self.neuron_config.tau
        self.v_threshold = self.neuron_config.v_threshold

        layers = []
        # Input layer
        input_layer = layer.Linear(in_dim, self.num_hidden_units[0], bias=False)
        layers.append(input_layer)
        layers.append(neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, 
                                     surrogate_function=surrogate.Sigmoid(), v_reset=None, 
                                     step_mode='s', backend='torch'))


        # Hidden layers
        for i in range(1, self.num_hidden_layers):
            in_features = self.num_hidden_units[i - 1]
            out_features = self.num_hidden_units[i]
            linear_layer = layer.Linear(in_features, out_features, bias=False)
            layers.append(linear_layer)
            if i in self.recurrent_layer:
                # Add a recurrent layer using ElementWiseRecurrentContainer
                recurrent_node = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, 
                                                surrogate_function=surrogate.Sigmoid(), v_reset=None,
                                                step_mode='s', backend='torch')
                recurrent_layer = layer.LinearRecurrentContainer(recurrent_node, out_features, out_features)
                layers.append(recurrent_layer)
            else:
                layers.append(neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, 
                                             surrogate_function=surrogate.Sigmoid(), v_reset=None, 
                                             step_mode='s', backend='torch'))

        # Output layer
        output_layer = layer.Linear(self.num_hidden_units[-1], out_dim, bias=False)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):  # T, B, F
        output_current = []
        for t in range(x.size(0)):
            output_current.append(self.network(x[t]))
        output_current = torch.stack(output_current, 0)
        output = output_current.sum(0)
        return output


class LSTMNet(nn.Module):
    def __init__(self, in_dim, out_dim, model_config):
        super().__init__()
        self.hidden_dim = 64

        self.input_layer = layer.Linear(in_dim, self.hidden_dim, bias=False)
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True)
        self.output_layer = layer.Linear(self.hidden_dim, out_dim, bias=False)

    def forward(self, x):  # T, B, F
        # x = x.permute(1, 0, 2)
        x = self.input_layer(x)
        x, _ = self.lstm(x)
        x = self.output_layer(x)
        output = x.sum(1)
        return output


class BaseNet_cupy(nn.Module):
    def __init__(self, in_dim, out_dim, model_config):
        super().__init__()
        self.network_config = model_config.network
        self.neuron_config = model_config.neuron

        self.num_hidden_units = self.network_config.num_hidden_units
        self.num_hidden_layers = self.network_config.num_hidden_layers
        self.recurrent_layer = self.network_config.recurrent_layer

        self.tau = self.neuron_config.tau
        self.v_threshold = self.neuron_config.v_threshold

        layers = []
        # Input layer
        input_layer = layer.Linear(in_dim, self.num_hidden_units[0], bias=False)
        layers.append(input_layer)
        layers.append(neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, surrogate_function=surrogate.ATan(), step_mode='m', backend='cupy'))


        # Hidden layers
        for i in range(1, self.num_hidden_layers):
            in_features = self.num_hidden_units[i - 1]
            out_features = self.num_hidden_units[i]
            linear_layer = layer.Linear(in_features, out_features, bias=False)
            layers.append(linear_layer)
            if i in self.recurrent_layer:
                # Add a recurrent layer using ElementWiseRecurrentContainer
                recurrent_node = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, surrogate_function=surrogate.ATan(), step_mode='s', backend='torch')
                recurrent_layer = layer.LinearRecurrentContainer(recurrent_node, out_features, out_features)
                layers.append(recurrent_layer)
            else:
                layers.append(neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, surrogate_function=surrogate.ATan(), step_mode='m', backend='cupy'))

        # Output layer
        output_layer = layer.Linear(self.num_hidden_units[-1], out_dim, bias=False)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):  # T, B, 
        return self.network(x)


if __name__ == '__main__':
    with open('./eg_conf.yaml', 'r') as file:
        config = yaml.safe_load(file)
    model_config = config['model']
    input_tensor = torch.rand(98, 16, 40).cuda()  # Example input tensor
    model = BaseNet(in_dim=40, out_dim=12, T=input_tensor.shape[0], model_config=model_config).cuda()
    print(model)
    print("Model parameter count = %.2f million" % (sum(param.numel() for param in model.parameters()) / 1e6))
    output = model(input_tensor)
    print(output.shape)