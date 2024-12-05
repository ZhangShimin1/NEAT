'''
Descripttion: Network architectures
version: 
Author: Shimin Zhang
Date: 2024-11-23 13:41:44
'''
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate

class SMLP_eg(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
        layer.Linear(input_dim, hidden_dim, bias=False),
        neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(),step_mode='m', backend='cupy'),
        layer.Linear(hidden_dim, hidden_dim, bias=False),
        neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(),step_mode='m', backend='cupy'),
        layer.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self,x):
        x = self.network(x)
        return x

# Model test
if __name__ == '__main__':
    input = torch.rand(98, 32, 40).cuda() # 4-second visual frames
    model = SMLP_eg(40, 64, 35).cuda()
    print("Model para number = %.2f"%(sum(param.numel() for param in model.parameters()) / 1e6))
    out = model(input)
    print(out.shape)
