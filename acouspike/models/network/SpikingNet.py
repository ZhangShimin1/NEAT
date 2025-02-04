import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from functools import partial
from acouspike.models.network.utils import reset_states
from acouspike.models.surrogate.surrogate import SurrogateGradient
from acouspike.models.neuron.lif import RLIF, LTC, CELIF, PMSN, SPSN, DHSNN, CLIF, adLIF
from acouspike.models.layers.layer import BatchNorm1d, ThresholdDependentBatchNorm1d, TemporalEffectiveBatchNorm1d

class SpikingNet(nn.Module):
    def __init__(self, input_size, 
                 hidden_size, 
                 num_layers, 
                 bidirectional=False, 
                 dropout=0.0, 
                 batch_first=True, 
                 spiking_neuron_name=None, 
                 recurrent=False, 
                 bn=None,
                 surrogate='triangle', 
                 alpha=1.0, 
                 decay=0.5,
                 threshold=0.5,
                 time_window = 512,
                 beta=0.1,
                 k=32,
                 branch=4):
        super(SpikingNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if spiking_neuron_name == 'lif':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(RLIF,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent,
                                    learning_rule='stbp',
                                    )
            
        elif spiking_neuron_name == 'ltc':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            b_j0 = beta
            spiking_neuron = partial(LTC,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent,
                                    b_j0=b_j0
                                    )
        elif spiking_neuron_name == 'celif':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            beta = beta
            spiking_neuron = partial(CELIF,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent,
                                    beta=beta
                                    )
        elif spiking_neuron_name == 'spsn':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(SPSN,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent,
                                    k=k
                                    )
            
        elif spiking_neuron_name == 'pmsn':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(PMSN,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent
                                    )
        elif spiking_neuron_name == 'dhsnn':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(DHSNN,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent,
                                    input_features=hidden_size,
                                    neuron_num=hidden_size,
                                    branch=branch
                                    )
            
        elif spiking_neuron_name == 'clif':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(CLIF,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent
                                    )
            
        elif spiking_neuron_name == 'adlif':
            surro_grad = SurrogateGradient(func_name=surrogate, a=alpha)
            exec_mode = "serial"
            spiking_neuron = partial(adLIF,
                                    decay=decay,
                                    threshold=threshold,
                                    time_step=time_window,
                                    surro_grad=surro_grad,
                                    exec_mode=exec_mode,
                                    recurrent=recurrent,
                                    input_features=hidden_size,
                                    neuron_num=hidden_size
                                    )
        else:
            print(f"{spiking_neuron_name}")
            raise NotImplementedError

        
        if bn == 'bn':
            self.bns = nn.ModuleList([BatchNorm1d(hidden_size[l]) for l in range(num_layers)])
        elif bn == 'ln':
            self.bns = nn.ModuleList([nn.LayerNorm(hidden_size[l]) for l in range(num_layers)])
        elif bn == 'tdbn':
            self.bns = nn.ModuleList([ThresholdDependentBatchNorm1d(alpha=1., v_th=spiking_neuron().threshold, num_features=hidden_size[l])
                   for l in range(num_layers)])
        elif bn == 'tebn':
            self.bns = nn.ModuleList([TemporalEffectiveBatchNorm1d(T=spiking_neuron().time_step, num_features=hidden_size[l]) for l in
                   range(num_layers)])
        elif bn is None:
            self.bns = None
        else:
            print(f"{bn} not implemented, BatchNorm type must be one of bn, ln, tdbn, tebn")
            raise NotImplementedError


        for hidden_layer_i in range(num_layers):
            exec("self.fc" + str(hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size)")
            if hidden_layer_i == (num_layers - 1):
                exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size, recurrent=False)")
            else:
                exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size)")
            input_size = hidden_size
        
        if spiking_neuron_name == 'celif':
            self.celif_TE = nn.Parameter(torch.zeros(hidden_size, time_window))
            nn.init.normal_(self.celif_TE, 0.01, 0.01)
            for hidden_layer_i in range(num_layers):
                exec("self.spk" + str(hidden_layer_i) + " .TE = self.celif_TE".format(hidden_layer_i))
                

        
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
        
        reset_states(self)
        for hidden_layer_i in range(0, self.num_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            if self.bns is not None:
                x = self.bns[hidden_layer_i](x)
            x = eval("self.spk" + str(hidden_layer_i))(x)

        output = x
        
        if self.batch_first:
            output = output.transpose(0, 1)  # Convert to (batch_size, seq_len, num_directions * hidden_size)
        
        if is_packed:
            output = pack_padded_sequence(output, lengths, batch_first=self.batch_first)
        
        # print(f"h_n: {h_n.size()}")

        return output, (None, None)
    
