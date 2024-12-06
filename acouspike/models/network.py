'''
Description: Baseline network configurations
version: 
Author: Shimin Zhang
Date: 2024-11-23 13:41:44
'''
import yaml

import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron, surrogate

@dataclass
class NeuronArgs(Serializable):
    tau: float = 2.
    v_threshold: float = 1.

class BaseNet(nn.Module):
    def __init__(self, in_dim, out_dim, model_config):
        super().__init__()
        self.network_config = model_config["network"]
        self.neuron_config = model_config["neuron"]

        self.num_hidden_units = self.network_config["num_hidden_units"]
        self.num_hidden_layers = self.network_config["num_hidden_layers"]
        self.recurrent_layer = self.network_config["recurrent_layer"]

        self.tau = self.neuron_config["tau"]
        self.v_threshold = self.neuron_config["v_threshold"]

        layers = []
        # Input layer
        input_layer = layer.Linear(in_dim, self.num_hidden_units[0], bias=False)
        layers.append(input_layer)
        layers.append(neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, surrogate_function=surrogate.ATan(), step_mode='s', backend='torch'))


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
                layers.append(neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, surrogate_function=surrogate.ATan(), step_mode='s', backend='torch'))

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


if __name__ == '__main__':
    with open('../../recipes/KWS/conf/default.yaml', 'r') as file:
        configuration = yaml.safe_load(file)
    model_config = configuration["model"]
    input_tensor = torch.rand(98, 16, 40).cuda()  # Example input tensor
    model = BaseNet(in_dim=40, out_dim=35, model_config=model_config).cuda()
    print(model)
    print("Model parameter count = %.2f million" % (sum(param.numel() for param in model.parameters()) / 1e6))
    output = model(input_tensor)
    print(output.shape)