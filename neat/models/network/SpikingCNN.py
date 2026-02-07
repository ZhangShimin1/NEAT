import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from functools import partial
from neat.models.network.utils import reset_states
from neat.models.surrogate.surrogate import TriangleSurroGrad
from neat.models.neuron.neuron import LIFLayer
from torch.nn import (
    Module,
    Conv2d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Flatten,
    Linear,
    BatchNorm2d,
    GroupNorm,
)
from torch.nn import Sequential


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class tdLayer(nn.Module):
    # todo: check code
    def __init__(self, layer, nb_steps):
        super(tdLayer, self).__init__()
        self.nb_steps = nb_steps
        self.layer = layer

    # def forward(self, x):
    #     # print('hello 1')
    #     out = []
    #     for step in range(self.nb_steps):
    #         out.append(self.layer(x[step]))
    #     return torch.stack(out)

    def forward(self, x):
        x = x.contiguous()
        x = self.layer(x.view(-1, *x.shape[2:]))
        return x.view(self.nb_steps, -1, *x.shape[1:])


def warpBN(channel, nb_steps):
    return tdLayer(nn.BatchNorm2d(channel), nb_steps)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, expand=1, **kwargs_spikes):
        super(BasicBlock, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes["nb_steps"]
        self.expand = expand
        self.conv1 = tdLayer(
            nn.Conv2d(
                in_planes,
                planes * expand,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            self.nb_steps,
        )
        self.bn1 = warpBN(planes * expand, self.nb_steps)
        self.spike1 = LIFLayer(**kwargs_spikes)
        self.conv2 = tdLayer(
            nn.Conv2d(
                planes, planes * expand, kernel_size=3, stride=1, padding=1, bias=False
            ),
            self.nb_steps,
        )
        self.bn2 = warpBN(planes * expand, self.nb_steps)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                tdLayer(
                    nn.Conv2d(
                        in_planes,
                        planes * self.expansion * expand,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    self.nb_steps,
                ),
                warpBN(self.expansion * planes * expand, self.nb_steps),
                # tdBatchNorm(nn.BatchNorm2d(planes * BasicBlock.expansion), alpha=1 / math.sqrt(2.))
            )
        self.spike2 = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # if self.expand != 1 and len(self.shortcut) == 0:
        #     x = torch.cat([x, x], dim=2)
        #     # todo: check here
        out += self.shortcut(x)
        out = self.spike2(out)
        return out


class SpikingResNet(nn.Module):
    def __init__(
        self, block, num_block_layers, num_classes=10, in_channel=1, **kwargs_spikes
    ):
        super(SpikingResNet, self).__init__()
        self.in_planes = 64
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes["nb_steps"]
        self.conv0 = nn.Sequential(
            tdLayer(
                nn.Conv2d(
                    in_channel,
                    self.in_planes,
                    kernel_size=7,
                    padding=3,
                    stride=2,
                    bias=False,
                ),
                nb_steps=self.nb_steps,
            ),
            warpBN(self.in_planes, self.nb_steps),
            LIFLayer(**kwargs_spikes),
        )
        self.layer1 = self._make_layer(block, 64, num_block_layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_block_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block_layers[3], stride=2)

        self.avg_pool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)), nb_steps=self.nb_steps)
        self.classifier = tdLayer(
            nn.Linear(512 * block.expansion, num_classes), nb_steps=self.nb_steps
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **self.kwargs_spikes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
        out = self.conv0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.classifier(out)
        return out.squeeze(1).permute(1, 0, 2)


# User interfaces
def spiking_resnet18(num_classes, in_channel=1, **kwargs_spikes):
    return SpikingResNet(
        block=BasicBlock,
        num_block_layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channel=in_channel,
        **kwargs_spikes,
    )


if __name__ == "__main__":
    kwargs_spikes = {
        "nb_steps": 4,
        "threshold": 1.0,
        "decay": 1.0,
        "surrogate_function": TriangleSurroGrad.apply,
    }
    input_tensor = torch.rand([32, 99, 40])
    model = spiking_resnet18(num_classes=10, in_channel=1, **kwargs_spikes)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
