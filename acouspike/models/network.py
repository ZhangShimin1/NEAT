'''
Descripttion: Network architectures
version: 
Author: Shimin Zhang
Date: 2024-11-23 13:41:44
'''
import torch
import torch.nn as nn
from dataclasses import dataclass
from simple_parsing import Serializable
from spikingjelly.activation_based import layer, neuron, surrogate

@dataclass
class ModelArgs(Serializable):
    input_dim: int = 40
    output_dim: int = 35
    num_hidden_layers: int = 3
    num_hidden_units: list = (64, 256, 128)  # Size of each hidden layer
    recurrent_layer: list = (0, 1)  # Index of layers that will be recurrent


class SMLP_eg(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        layers = []

        # Input layer
        input_layer = layer.Linear(args.input_dim, args.num_hidden_units[0], bias=False)
        layers.append(input_layer)
        layers.append(neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), step_mode='m', backend='cupy'))

        # Hidden layers
        for i in range(1, args.num_hidden_layers):
            in_features = args.num_hidden_units[i - 1]
            out_features = args.num_hidden_units[i]
            linear_layer = layer.Linear(in_features, out_features, bias=False)
            layers.append(linear_layer)
            if i in args.recurrent_layer:
                # Add a recurrent layer using ElementWiseRecurrentContainer
                recurrent_node = neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), step_mode='s', backend='torch')
                recurrent_layer = layer.LinearRecurrentContainer(recurrent_node, out_features, out_features)
                layers.append(recurrent_layer)
            else:
                layers.append(neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), step_mode='m', backend='cupy'))

        # Output layer
        output_layer = layer.Linear(args.num_hidden_units[-1], args.output_dim, bias=False)
        layers.append(output_layer)

        # Define the entire network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x

# Model test
if __name__ == '__main__':
    args = ModelArgs()
    input_tensor = torch.rand(98, 32, args.input_dim).cuda()  # Example input tensor
    model = SMLP_eg(args).cuda()
    print("Model parameter count = %.2f million" % (sum(param.numel() for param in model.parameters()) / 1e6))
    output = model(input_tensor)
    print(output.shape)
