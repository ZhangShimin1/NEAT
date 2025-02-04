import torch
import torch.nn as nn
import sys
sys.path.append('/home/zysong/AcouSpike/acouspike')
from models.SNN import Spiking_LSTM, TCN, SpikingNet, SpkTransformerNet, SSMNet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.SNN.utils import reset_backward_rnn_state

from simple_parsing.helpers import Serializable
from dataclasses import dataclass, field

@dataclass
class ModelWrapperArgs(Serializable):
    model_name: str
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    dropout: float = 0.0
    neuron_type: str = "spiking_lstm"
    bidirectional: bool = False
    batch_first: bool = True
    bn: str = None
    neuron_args: dict = field(default_factory=dict)
    SG_args: dict = field(default_factory=dict) 

class ModelWrapper(nn.Module):
    """
    A wrapper class for various spiking neural network models.
    """
    
    AVAILABLE_MODELS = {
        'spiking_lstm': 'Spiking LSTM model',
        'tcn': 'Temporal Convolutional Network',
        'spikeformer': 'Spiking Transformer',
        'ssm': 'State Space Model',
        'spikingnet': 'Basic Spiking Neural Network'
    }

    def __init__(
        self,
        model_name: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float = 0.0,
        neuron_type: str = "spiking_lstm",
        bidirectional: bool = False,
        bn: str = None,
        batch_first: bool = True,
        surrogate: str = 'triangle',
        alpha: float = 1.0,
        decay: float = 0.5,
        threshold: float = 0.5,
        time_window: int = 512,
        beta: float = 0.1,
        k: int = 32,
        ksize: int = 7,
        lr: float = 0.1,
        nhead: int = 4,
        recurrent: bool = False,
        branch: int = 4,
        **kwargs
    ):
        """
        Initialize the model wrapper.

        Args:
            model_name (str): Name of the model to use
            input_size (int): Size of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of layers
            neuron_type (str): Type of neuron
            bidirectional (bool): Whether to use bidirectional model
            dropout (float): Dropout rate
            batch_first (bool): Whether batch dimension is first
            surrogate (str): Surrogate function type
            alpha (float): Alpha parameter for neuron
            decay (float): Decay rate for neuron
            threshold (float): Firing threshold
            time_window (int): Time window for processing
            beta (float): Beta parameter
            k (int): K parameter for TCN
            ksize (int): Kernel size for TCN
            lr (float): Learning rate for some models
            nhead (int): Number of attention heads for transformer
            recurrent (bool): Whether to use recurrent connections
            branch (int): Number of branches for DHSNN
            **kwargs: Additional arguments
        """
        super().__init__() 
        print(f"Initializing {model_name} model...")    
        self.model_name = model_name.lower()
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )

        # Initialize the selected model based on type
        if self.model_name == 'spiking_lstm':
            self.model = Spiking_LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout
            )
        elif self.model_name == 'tcn':
            self.model = TCN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=batch_first,
                decay=decay,
                threshold=threshold,
                ksize=ksize
            )
        elif self.model_name == 'spikeformer':
            self.model = SpkTransformerNet(
                input_size=input_size,
                hidden_size=hidden_size,
                nhead=nhead,
                num_hidden_layers=num_layers,
                dropout=dropout,
                surrogate=surrogate,
                alpha=alpha,
                decay=decay,
                threshold=threshold,
                recurrent=recurrent,
                time_window=time_window,
                T=1
            )
        elif self.model_name == 'ssm':
            self.model = SSMNet(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=batch_first,
                spiking_neuron_name=neuron_type,
                recurrent=recurrent,
                surrogate=surrogate,
                alpha=alpha,
                decay=decay,
                threshold=threshold,
                time_window=time_window,
                lr=lr
            )
        else:  # spikingnet
            self.model = SpikingNet(input_size=input_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    bidirectional=bidirectional, 
                                    dropout=dropout, 
                                    batch_first=batch_first, 
                                    spiking_neuron_name=neuron_type, 
                                    recurrent=recurrent,
                                    bn=bn,
                                    surrogate=surrogate, 
                                    alpha=alpha, 
                                    decay=decay,
                                    threshold=threshold,
                                    time_window =time_window,
                                    beta=beta,
                                    k=k,
                                    branch=branch)
        if bidirectional:
            self.l_last = torch.nn.Linear(hidden_size * 2, output_size)
        else:
            self.l_last = torch.nn.Linear(hidden_size, output_size)
        self.typ = neuron_type

    def forward(self, inputs):
        """RNN forward

        :param torch.Tensor inputs: batch of padded input sequences (B, Tmax, D)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        ys, states = self.model(inputs)
        projected = torch.tanh(
            self.l_last(ys.contiguous().view(-1, ys.size(2)))
        )
        xs_pad = projected.view(ys.size(0), ys.size(1), -1)
        return xs_pad, states  # x: utt list of frame x dim

def test_model_wrapper():
    """
    Test function to verify ModelWrapper functionality using ModelWrapperArgs.
    """
    # Base parameters common to all models
    base_args = {
        'input_size': 128,
        'hidden_size': 256,
        'output_size': 10,
        'num_layers': 2,
        'dropout': 0.1,
        'bidirectional': False,
        'batch_first': True,
    }

    # Model-specific configurations
    model_configs = {
        'spiking_lstm': ModelWrapperArgs(
            model_name='spiking_lstm',
            **base_args,
            neuron_type='spiking_lstm',
            neuron_args={},
            SG_args={}
        ),
        'tcn': ModelWrapperArgs(
            model_name='tcn',
            **base_args,
            neuron_type='spkbinaryssm',
            neuron_args={
                'ksize': 7,
                'decay': 0.5,
                'threshold': 0.5,
            },
            SG_args={
                'surrogate_type': 'triangle',
                'alpha': 2.0
            }
        ),
        'spikeformer': ModelWrapperArgs(
            model_name='spikeformer',
            **base_args,
            neuron_type='spkbinaryssm',
            neuron_args={
                'nhead': 4,
                'recurrent': False,
                'time_window': 512,
                'decay': 0.5,
                'threshold': 0.5,
                'T': 1
            },
            SG_args={
                'surrogate_type': 'triangle',
                'alpha': 2.0
            }
        ),
        'ssm': ModelWrapperArgs(
            model_name='ssm',
            **base_args,
            neuron_type='spkbinaryssm',
            neuron_args={
                'recurrent': False,
                'time_window': 512,
                'decay': 0.5,
                'threshold': 0.5,
                'lr': 0.1
            },
            SG_args={
                'surrogate_type': 'triangle',
                'alpha': 2.0
            }
        ),
        'spikingnet': ModelWrapperArgs(
            model_name='spikingnet',
            **base_args,
            neuron_type='celif',
            neuron_args={
                'recurrent': False,
                'time_window': 512,
                'decay': 0.5,
                'threshold': 0.5,
                'beta': 0.1,
                'branch': 4
            },
            SG_args={
                'surrogate_type': 'triangle',
                'alpha': 2.0
            }
        )
    }

    def run_model_tests(model_args):
        print(f"\nTesting {model_args.model_name}...")
        try:
            # Test initialization
            model = ModelWrapper(
                model_name=model_args.model_name,
                input_size=model_args.input_size,
                hidden_size=model_args.hidden_size,
                output_size=model_args.output_size,
                num_layers=model_args.num_layers,
                dropout=model_args.dropout,
                neuron_type=model_args.neuron_type,
                bidirectional=model_args.bidirectional,
                batch_first=model_args.batch_first,
                **model_args.neuron_args,
                **model_args.SG_args
            )
            print("✓ Model initialization successful")

            # Test forward pass
            batch_size = 32
            seq_length = 100
            x = torch.randn(batch_size, seq_length, model_args.input_size)
            xs_pad, states = model(x)
            print(f"✓ Forward pass successful. Output shape: {xs_pad.shape}")

            # Test device movement
            if torch.cuda.is_available():
                model.to('cuda')
                print("✓ Device movement successful")
            else:
                print("- Skipping device test (CUDA not available)")

            # Test training mode
            model.train()
            assert model.model.training == True
            model.eval()
            assert model.model.training == False
            print("✓ Training mode switches successful")

        except Exception as e:
            print(f"✗ Test failed for {model_args.model_name}: {str(e)}")

    # Run tests for all models
    for model_name in model_configs:
        run_model_tests(model_configs[model_name])

if __name__ == "__main__":
    test_model_wrapper()