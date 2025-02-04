import torch
import sys
sys.path.insert(0, "/home/zysong/AcouSpike/acouspike")
from models.SNN import Spiking_LSTM, TCN, SpikingNet, SpkTransformerNet, SSMNet
from models.SNN.utils import count_parameters
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SNN_encoder(torch.nn.Module):
    """RNN module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, neuron_type="spiking_lstm", recurrent=False, surrogate='triangle', alpha=1.0, decay=0.5, threshold=0.5, time_window=512, beta=0.1, k=32, ksize=7, lr=0.1, nhead=4):
        super(SNN_encoder, self).__init__()
        bidir = neuron_type[0] == "b"

        
        if neuron_type == 'spiking_lstm':
            self.nbrnn = Spiking_LSTM(idim, cdim, elayers, bidirectional=bidir, dropout=dropout)
        elif neuron_type == 'tcn':
            self.nbrnn = TCN(input_size=idim, 
                        hidden_size=cdim, 
                        num_layers=elayers, 
                        bidirectional=bidir, 
                        dropout=0.0, 
                        batch_first=True, 
                        spiking_neuron_name=neuron_type, 
                        recurrent=recurrent,
                        surrogate=surrogate, 
                        alpha=alpha, 
                        decay=decay,
                        threshold=threshold,
                        time_window =time_window,
                        beta=beta,
                        k=k,
                        ksize=ksize)
            
        elif neuron_type == 'spktransformer':
            self.nbrnn = SpkTransformerNet(
                        input_size=idim,
                        hidden_size=cdim,
                        nhead=nhead,
                        num_hidden_layers=elayers,
                        dropout=0,
                        surrogate=surrogate,
                        alpha=alpha,
                        decay=decay,
                        threshold=threshold,
                        recurrent=False,
                        time_window=512,
                        T=1,
                        )
        elif neuron_type in ['spkbinaryssm', 'gsussm']:
            self.nbrnn = SSMNet(input_size=idim, 
                        hidden_size=cdim, 
                        num_layers=elayers, 
                        bidirectional=bidir, 
                        dropout=0.0, 
                        batch_first=True, 
                        spiking_neuron_name=neuron_type, 
                        recurrent=recurrent,
                        surrogate=surrogate, 
                        alpha=alpha, 
                        decay=decay,
                        threshold=threshold,
                        time_window =time_window,
                        lr=lr
                        )
        else:
            self.nbrnn = SpikingNet(input_size=idim, 
                                    hidden_size=cdim, 
                                    num_layers=elayers, 
                                    bidirectional=bidir, 
                                    dropout=0.0, 
                                    batch_first=True, 
                                    spiking_neuron_name=neuron_type, 
                                    recurrent=recurrent,
                                    surrogate=surrogate, 
                                    alpha=alpha, 
                                    decay=decay,
                                    threshold=threshold,
                                    time_window =time_window,
                                    beta=beta,
                                    k=k)
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)
        self.typ = neuron_type
        para = count_parameters(self.nbrnn)
        # logging.info(f"Parameter number: {para}")

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        # logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs_pack = pack_padded_sequence(xs_pad, ilens.cpu(), batch_first=True)
        if self.training:
            self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed,
            # it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state
            # (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hidden=prev_state)
        # print(f"xs_pack: {xs_pack.size()}, prev_state: {prev_state.size()}, states: {states.size()}, yes: {ys.size()}")
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(
            self.l_last(ys_pad.contiguous().view(-1, ys_pad.size(2)))
        )
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens, states  # x: utt list of frame x dim


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes

    Useful in processing of sliding windows over the inputs
    """
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.0
    else:
        states[1::2] = 0.0
    return states
