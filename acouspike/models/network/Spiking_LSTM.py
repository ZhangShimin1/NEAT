import math
import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from acouspike.models.surrogate.surrogate import TriangleSurroGrad

class Spiking_LSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Spiking_LSTM_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold=0.1

        # Define the gates
        self.i2h = nn.Linear(input_size, 2 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size)
        self.surrogate_function=TriangleSurroGrad.apply
        # self.surrogate_function=MultiSpike(dim=hidden_size, T=6)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        # Gates computation
        gates = self.i2h(x) + self.h2h(h_prev)
        f_gate, c_tilde = gates.chunk(2, 1)

        f_gate = torch.sigmoid(f_gate)

        # Cell state
        c_next = f_gate * c_prev + (1-f_gate) * c_tilde

        # #Spiking neuron update
        # h_next = self.surrogate_function(c_next - self.threshold) #Surrogate 
        # # Hidden state
        # c_next = c_next - c_next * h_next

        #Q_trick
        h_next = self.surrogate_function(c_next.unsqueeze(0)).squeeze(0)

        return h_next, c_next


class Spiking_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, dropout=0.0, batch_first=True):
        super(Spiking_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = batch_first

        self.cells = nn.ModuleList([Spiking_LSTM_Cell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        if bidirectional:
            self.rev_cells = nn.ModuleList([Spiking_LSTM_Cell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def flatten_parameters(self):
        # Ensure all parameters are contiguous in memory for efficiency
        for cell in self.cells:
            for param in cell.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
        if self.bidirectional:
            for cell in self.rev_cells:
                for param in cell.parameters():
                    if not param.is_contiguous():
                        param.data = param.data.contiguous()

    def forward(self, x, hidden=None):
        is_packed = isinstance(x, PackedSequence)
        assert hidden is None, "Not allowed to pass previous states in the current imple."
        if is_packed:
            x, lengths = pad_packed_sequence(x, batch_first=self.batch_first)
        # print(f"x: {x.size()}") # [90, 101, 2560]
        # print(f"max lengths: {lengths.max()}") # [90]
        # if isinstance(hidden, tuple):
        #     print(f"tuple hidden: {len(hidden)} {hidden[0].size()}")
        # elif hidden is None:
        #     print(f"hidden: {hidden}")
        # else:
        #     print(f"hidden: {hidden.size()}")
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
            x_fwd = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_size)
        else:
            seq_len, batch_size, _ = x.size()
            x_fwd = x
        
        x_rev = x_fwd.flip(0)  # Reverse the input sequence # [101, 90, 2560]
        # print(f"x_rev: {x_rev.size()}")
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h_0, c_0 = hidden
        
        h_n, c_n = [], []
        output_fwd = []
        output_rev = []
        for i in range(self.num_layers):
            h_i, c_i = h_0[i], c_0[i]
            h_t = []

            for t in range(seq_len):
                h_i, c_i = self.cells[i](x_fwd[t], (h_i, c_i))
                h_t.append(h_i)

            h_t = torch.stack(h_t, dim=0)
            if i < self.num_layers - 1:
                h_t = self.dropout_layer(h_t)
            x_fwd = h_t
            h_n.append(h_i)
            c_n.append(c_i)
            output_fwd.append(h_t)
        out_fwd = h_t

        # Reverse LSTM
        if self.bidirectional:
            h_n_rev, c_n_rev = [], []

            for i in range(self.num_layers):
                h_i, c_i = h_0[i], c_0[i]
                h_t = []

                for t in range(seq_len):
                    h_i, c_i = self.rev_cells[i](x_rev[t], (h_i, c_i))
                    h_t.append(h_i)

                h_t = torch.stack(h_t, dim=0)
                h_t = h_t.flip(0)  # Reverse the output sequence
                if i < self.num_layers - 1:
                    h_t = self.dropout_layer(h_t)
                x_rev = h_t    
                h_n_rev.append(h_i)
                c_n_rev.append(c_i)
                output_rev.append(h_t)
            out_rev = h_t
            output = torch.cat([out_fwd, out_rev], dim=-1)
            h_n = torch.cat([torch.stack(h_n, dim=0), torch.stack(h_n_rev, dim=0)], dim=0)
            c_n = torch.cat([torch.stack(c_n, dim=0), torch.stack(c_n_rev, dim=0)], dim=0)

            
            if self.batch_first:
                output = output.transpose(0, 1)  # Convert to (batch_size, seq_len, num_directions * hidden_size)

            if is_packed:
                output = pack_padded_sequence(output, lengths, batch_first=self.batch_first)


            return output, (h_n, c_n)

        output = out_fwd
        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)
        # print(f"output: {output.size()}")
        if self.batch_first:
            output = output.transpose(0, 1)  # Convert to (batch_size, seq_len, num_directions * hidden_size)
        
        if is_packed:
            output = pack_padded_sequence(output, lengths, batch_first=self.batch_first)
        
        # print(f"h_n: {h_n.size()}")

        return output, (h_n, c_n)
