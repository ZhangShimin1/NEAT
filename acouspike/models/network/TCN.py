import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from functools import partial
from acouspike.models.network.utils import reset_states
from acouspike.models.neuron.neuron import LIFNode
from acouspike.models.surrogate.surrogate import TriangleSurroGrad

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, spiking_neuron=None):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU() if spiking_neuron is None else spiking_neuron()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU() if spiking_neuron is None else spiking_neuron()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU() if spiking_neuron is None else nn.Identity()
        self.init_weights()
        if spiking_neuron is not None and self.downsample is not None:
            self.downsample = nn.Sequential(self.downsample, spiking_neuron())

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x): # [B, N, T]
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, spiking_neuron=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, spiking_neuron=spiking_neuron)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, 
                 hidden_size, 
                 num_layers, 
                 dropout=0.0, 
                 batch_first=True, 
                 decay=0.5,
                 threshold=0.5,
                 ksize=7):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        num_channels = [hidden_size] * num_layers

        spiking_neuron = partial(LIFNode,
                            decay_factor=decay,
                            threshold=threshold,
                            surrogate_function=TriangleSurroGrad.apply,
                            hard_reset=True,
                            detach_reset=False,
                            detach_mem=False,
                            )
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=ksize, dropout=dropout,
                                   spiking_neuron=spiking_neuron)
                

        
    def flatten_parameters(self):
        # Ensure all parameters are contiguous in memory for efficiency
        pass

    def forward(self, x, hidden=None):
        is_packed = isinstance(x, PackedSequence)
        assert hidden is None, "Not allowed to pass previous states in the current imple."
        if is_packed:
            x, lengths = pad_packed_sequence(x, batch_first=self.batch_first)
        # print(f"x: {x.size()}") # [90, 101, 2560]
        # print(f"max lengths: {lengths.max()}") # [90]
        
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_size)
        else:
            seq_len, batch_size, _ = x.size()
            x = x     
        x = x.permute(1, 2, 0).contiguous()   
        
        reset_states(self)
        T = 1
        for t in range(T):
            y1 = self.tcn(x)
            y1 = y1.permute(2, 0, 1).contiguous()

        output = y1
        
        if self.batch_first:
            output = output.transpose(0, 1)  # Convert to (batch_size, seq_len, num_directions * hidden_size)
        
        if is_packed:
            output = pack_padded_sequence(output, lengths, batch_first=self.batch_first)
        
        # print(f"h_n: {h_n.size()}")

        return output, (None, None)
    
