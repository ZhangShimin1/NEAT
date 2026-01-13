import torch
import math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from functools import partial
from acouspike.models.network.utils import reset_states
from acouspike.models.surrogate.surrogate import SurrogateGradient
from acouspike.models.neuron.S4DModule import S4D


class SSMNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional=False,
        dropout=0.0,
        batch_first=True,
        spiking_neuron_name=None,
        recurrent=False,
        surrogate="triangle",
        alpha=1.0,
        decay=0.5,
        threshold=0.5,
        time_window=512,
        lr=0.1,
    ):
        super(SSMNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if spiking_neuron_name == "spkbinaryssm":
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(
                S4D,
                dropout=0.1,
                lr=min(0.001, lr),
                binary="binary",
                threshold=0.0,
                time_step=time_window,
                surro_grad=surro_grad,
            )

        elif spiking_neuron_name == "gsussm":
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(S4D, dropout=0.1, lr=min(0.001, lr), binary="GSU")
        else:
            print(f"{spiking_neuron_name}")
            raise NotImplementedError

        for hidden_layer_i in range(num_layers):
            exec(
                "self.spk"
                + str(hidden_layer_i)
                + " = spiking_neuron(neuron_num=hidden_size)"
            )
            if hidden_layer_i == 0:
                exec(
                    "self.fc"
                    + str(hidden_layer_i)
                    + " = nn.Linear(in_features=input_size, out_features=hidden_size)"
                )
            input_size = hidden_size

        self.max_length = 0

    def flatten_parameters(self):
        # Ensure all parameters are contiguous in memory for efficiency
        pass

    def forward(self, x, hidden=None):
        is_packed = isinstance(x, PackedSequence)
        assert hidden is None, (
            "Not allowed to pass previous states in the current imple."
        )
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
        self.max_length = max(self.max_length, seq_len)
        # logging.info(f"max_length: {self.max_length}")
        reset_states(self)
        for hidden_layer_i in range(0, self.num_layers):
            if hidden_layer_i == 0:
                x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)

        output = x

        if self.batch_first:
            output = output.transpose(
                0, 1
            )  # Convert to (batch_size, seq_len, num_directions * hidden_size)

        if is_packed:
            output = pack_padded_sequence(output, lengths, batch_first=self.batch_first)

        # print(f"h_n: {h_n.size()}")

        return output, (None, None)
