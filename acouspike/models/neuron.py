
import math
import inspect
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

from .sg import SurrogateGradient as SG


class BaseNeuron(nn.Module):
    def __init__(self, exec_mode: str="serial"):
        super(BaseNeuron, self).__init__()
        self.exec_mode    = exec_mode
        self._exec_config = {
            "default": self._serial_process,
            "serial" : self._serial_process,
            "fused"  : self._temporal_fused_process,
        }

    def forward(self, tx, v=None):
        execution_proc = self._exec_config.get(self.exec_mode)
        if execution_proc is not None:
            return execution_proc(tx, v)
        else:
            raise ValueError("Invalid `execution_mode`.")

    def _serial_process(self, _):
        raise NotImplementedError(f"The `{inspect.currentframe().f_code.co_name}` method of the subclass `{type(self).__name__}` needs to be implemented.")

    def _temporal_fused_process(self, _):
        raise NotImplementedError(f"The `{inspect.currentframe().f_code.co_name}` method of the subclass `{type(self).__name__}` needs to be implemented.")

class LIFAct(Function):
    @staticmethod
    def forward(ctx, v, rest, decay, threshold, time_step, surro_grad):
        ctx.save_for_backward(v)
        ctx.rest = rest
        ctx.decay = decay
        ctx.threshold = threshold
        ctx.time_step = time_step
        ctx.surro_grad = surro_grad
        return v.gt(threshold).float()

    @staticmethod
    def backward(ctx, grad_y):
        (v,) = ctx.saved_tensors
        grad_v = grad_y * ctx.surro_grad(
            v,
            rest=ctx.rest,
            decay=ctx.decay,
            threshold=ctx.threshold,
            time_step=ctx.time_step,
        )
        return grad_v, None, None, None, None, None


class LIFAct_thresh(Function):
    @staticmethod
    def forward(ctx, v, rest, decay, threshold, time_step, surro_grad):
        ctx.save_for_backward(v, threshold)
        ctx.rest = rest
        ctx.decay = decay
        ctx.time_step = time_step
        ctx.surro_grad = surro_grad
        return v.gt(threshold).float()

    @staticmethod
    def backward(ctx, grad_y):
        (v, threshold) = ctx.saved_tensors
        grad_v = grad_y * ctx.surro_grad(
            v,
            rest=ctx.rest,
            decay=ctx.decay,
            threshold=threshold,
            time_step=ctx.time_step,
        )
        return grad_v, None, None, -grad_v, None, None



class RLIF(BaseNeuron):
    """
        Recurrent spiking neural network, with the choice of BP methods

    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False,
            learning_rule: str = "stbp",
            truncated_t: int = 1000,
            bn=None,
            last_layer=False
    ):
        super(RLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.truncated_t = truncated_t
        self.learning_rule = learning_rule
        self.recurrent = recurrent
        self.bn = bn
        self.last_layer = last_layer
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)

    def __repr__(self):
        # TODO: Avoid type sensitivity caused by `self.surro_grad`
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}\", "
            f"learning_rule=\"{self.learning_rule}\", "
            f"truncated_t=\"{self.truncated_t}\", "
            f"batchnorm=\"{self.bn}\", "
            f"last_layer=\"{self.last_layer}\", "
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            return_state = False

        if self.recurrent and self.learning_rule == 'eprop':
            recurrent_trace = torch.zeros_like(tx[0])

        if self.bn is not None:
            tx = self.bn(tx)

        for t, x in enumerate(tx):
            if self.recurrent:
                if self.training and self.learning_rule == 'eprop':
                    recurrent_trace = self.decay * recurrent_trace.detach() + y.detach()
                    recurrent_trace_output = self.recurrent_weight(recurrent_trace.detach())
                    x = x + self.recurrent_weight(
                        y.detach()).detach() + recurrent_trace_output - recurrent_trace_output.detach()
                elif self.learning_rule in ['sltt']:
                    x = x + self.recurrent_weight(y.detach())
                elif self.learning_rule == 'tbptt':
                    if t % self.truncated_t == 0:
                        x = x + self.recurrent_weight(y.detach())
                    else:
                        x = x + self.recurrent_weight(y)
                else:
                    x = x + self.recurrent_weight(y)
            if self.learning_rule == 'stbp':
                v = self.decay * v + x
            elif self.learning_rule in ['sdbp', 'eprop', 'sltt']:
                v = self.decay * v.detach() + x
            elif self.learning_rule == 'notd':
                v = x
            elif self.learning_rule == 'tbptt':
                if t % self.truncated_t == 0:
                    v = v.detach()
                    y = y.detach()
                v = self.decay * v + x
            else:
                raise NotImplementedError
            # if not self.last_layer:
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            if self.learning_rule in ['sltt', 'eprop', 'sdbp']:
                v = v - v * y.detach() + self.rest * y.detach()  # Hard reset
                # print(f"t: {t} y: {y.sum()} y size: {y.size()}")
            elif self.learning_rule == 'notd':
                v = v
            else:
                v = v - v * y + self.rest * y  # Hard reset
            # else:
            #     y = v
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        raise NotImplementedError
        # else: # todo: add recurrent acceleration


class Recurrent_LIF(BaseNeuron):
    """
        Recurrent spiking neural network.
    """

    def __init__(
            self,
            rest: float = 0.0,

            decay: float = None,
            threshold: float = None,
            neuron_num: int = None,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(Recurrent_LIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)

    def __repr__(self):
        # TODO: Avoid type sensitivity caused by `self.surro_grad`
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            return_state = False
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v = self.decay * v * (1.0 - y) + self.rest * y + x
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        raise NotImplementedError
        # else: # todo: add recurrent acceleration


class PLIF(BaseNeuron):
    """
        Altered from Spikingjelly
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(PLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)

        init_w = - math.log(1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            return_state = False

        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v = self.w.sigmoid() * v * (1.0 - y) + self.rest * y + x
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        else:
            return torch.stack(ty)

    # def _temporal_fused_process(self, tx):
    # else: # todo: add recurrent acceleration


class ALIF(BaseNeuron):
    """
        Altered from https://github.com/byin-cwi/Efficient-spiking-networks
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(ALIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.tau_adp = nn.Parameter(torch.Tensor(self.neuron_num))
        self.tau_m = nn.Parameter(torch.Tensor(self.neuron_num))
        nn.init.normal_(self.tau_adp, 700, 25)
        nn.init.normal_(self.tau_m, 20, 5)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            b = state[2]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            b = 0.01
            return_state = False

        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v, y, thresh, b = self.mem_update_adp(x, v, y, self.tau_adp, self.tau_m, b)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y, b)
        else:
            return torch.stack(ty)

    def mem_update_adp(self, inputs, mem, spike, tau_adp, tau_m, b, dt=1, isAdapt=1):

        alpha = torch.exp(-1. * dt / tau_m).cuda()
        ro = torch.exp(-1. * dt / tau_adp).cuda()
        # tau_adp is tau_adaptative which is learnable # add requiregredients
        if isAdapt:
            beta = 1.8
        else:
            beta = 0.
        b = ro * b + (1 - ro) * spike
        # B = 0.01 + beta * b
        # mem = mem * alpha + (1 - alpha) * 1. * inputs - B * spike * dt # the orginal setting is hard to converge

        B = self.threshold + beta * b
        mem = mem * alpha + inputs - B * spike * dt
        spike = LIFAct_thresh.apply(mem, self.rest, self.decay, B, self.time_step, self.surro_grad)
        return mem, spike, B, b
    # def _temporal_fused_process(self, tx):
    # else: # todo: add recurrent acceleration


class GLIF(BaseNeuron):
    """
        Altered from https://github.com/Ikarosy/Gated-LIF
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(GLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)

        self.gate = [0.8, 0.2, 0.8]
        self.param = [0.25, 0.5, 0.5 / 8, 0.5]
        self.alpha, self.beta, self.gamma = [
            nn.Parameter(- math.log(1 / ((i - 0.5) * 0.5 + 0.5) - 1) * torch.ones(self.neuron_num, dtype=torch.float))
            for i in self.gate]

        self.tau, self.Vth, self.leak = [
            nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.neuron_num, dtype=torch.float))
            for i in self.param[:-1]]
        self.reVth = nn.Parameter(- math.log(1 / self.param[1] - 1) * torch.ones(self.neuron_num, dtype=torch.float))
        # t, c
        self.conduct = \
        [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.time_step, self.neuron_num), dtype=torch.float))
         for i in self.param[3:]][0]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            return_state = False

        step = 0
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v, y = self.extended_state_update(v, y, x, tau=self.tau.sigmoid(),
                                              Vth=self.Vth.sigmoid(),
                                              leak=self.leak.sigmoid(),
                                              conduct=self.conduct[step].sigmoid(),
                                              reVth=self.reVth.sigmoid())
            ty.append(y)
            step = step + 1
        if return_state:
            return torch.stack(ty), (v, y)
        else:
            return torch.stack(ty)

    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # [v: T B C]
        al, be, ga = self.alpha.view(1, -1).sigmoid(), self.beta.view(1, -1).sigmoid(), self.gamma.view(1, -1).sigmoid()
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :]))
        u_t_n1 = ((1 - al * (1 - tau[None, :])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :]) + \
                 I_t1 - (1 - ga) * reVth[None, :] * o_t_n1.clone()
        o_t_n1 = LIFAct_thresh.apply(u_t_n1, self.rest, self.decay, Vth[None, :], self.time_step, self.surro_grad)
        return u_t_n1, o_t_n1
    # def _temporal_fused_process(self, tx):
    # else: # todo: add recurrent acceleration


class CLIF(BaseNeuron):
    """
        Altered from https://github.com/HuuYuLong/Complementary-LIF
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(CLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)

        self.gamma = 0.5

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            u = state[0]
            y = state[1]
            m = state[2]
            return_state = True
        else:
            u = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            m = torch.zeros_like(tx[0])
            return_state = False

        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            u = self.gamma * u + x
            y = LIFAct.apply(u, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
            m = m * torch.sigmoid_((1. - self.gamma) * u) + y
            u = u - y * (self.threshold + torch.sigmoid_(m))
        if return_state:
            return torch.stack(ty), (u, y, m)
        else:
            return torch.stack(ty)
    # def _temporal_fused_process(self, tx):
    # else: # todo: add recurrent acceleration


class SPSN(BaseNeuron):
    """
        Altered from Spikingjelly
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(SPSN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.return_mem = False

        # self.register_memory('queue', [])
        self.k = 32
        self.backend = 'conv'
        self.thresh = torch.tensor([self.threshold]).cuda()

        weight = torch.ones([self.k])
        for i in range(self.k - 2, -1, -1):
            weight[i] = weight[i + 1] / 2.

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.as_tensor(-0.))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        step_num = tx.size(0)
        if isinstance(state, tuple):
            return_state = True
        else:
            return_state = False

        x_seq = tx.flatten(1).t().unsqueeze(1)
        x_seq = nn.functional.pad(x_seq, pad=(self.k - 1, 0))
        v = nn.functional.conv1d(x_seq, self.weight.view(1, 1, -1), stride=1)

        v = v.squeeze(1).t().contiguous().view(step_num, -1, self.neuron_num) + self.bias * self.thresh

        ty = LIFAct_thresh.apply(v, self.rest, self.decay, self.thresh, self.time_step, self.surro_grad)

        if return_state:
            return ty, (state)
        elif self.return_mem:
            return v[-1,].unsqueeze(0)
        else:
            return ty


class LTC(BaseNeuron):
    """
        Altered from https://github.com/byin-cwi/sFPTT/blob/main/fptt/fptt_mnist/snn_models_LIF4_save4.py
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False,
            b_j0: float = 0.2
    ):
        super(LTC, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.return_mem = False
        self.beta = 0.2
        self.b_j0 = b_j0
        # self.act1 = sigmoid_beta(is_train=True)
        # self.act2 = sigmoid_beta(is_train=True)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()
        self.layer1_tauM = nn.Linear(self.neuron_num * 2, self.neuron_num)
        self.layer1_tauAdp = nn.Linear(self.neuron_num * 2, self.neuron_num)
        nn.init.xavier_normal_(self.layer1_tauM.weight)
        nn.init.xavier_normal_(self.layer1_tauAdp.weight)
        nn.init.constant_(self.layer1_tauM.bias, 0)
        nn.init.constant_(self.layer1_tauAdp.bias, 0)

        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            mem = state[0]
            y = state[1]
            b = state[2]
            return_state = True
        else:
            mem = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            # b = self.b_j0 * torch.ones_like(tx[0])
            b = self.threshold * torch.ones_like(tx[0])
            return_state = False
        step = 0
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)

            alpha = self.act1(self.layer1_tauM(torch.cat((x, mem), dim=-1)))  # to avoid gradient explosion
            ro = self.act2(self.layer1_tauAdp(torch.cat((x, b), dim=-1)))
            beta = self.beta

            b = ro * b + (1 - ro) * y
            B = self.threshold + beta * b

            d_mem = - mem + x
            mem = mem + d_mem * alpha

            y = LIFAct_thresh.apply(mem, self.rest, self.decay, B, self.time_step, self.surro_grad)
            mem = (1 - y) * mem
            ty.append(y)
            step = step + 1
        # print(self.neuron_num,torch.stack(ty)[0].mean().item(),torch.stack(ty)[-1].mean().item(), tx[0].mean().item(), tx[-1].mean().item(), mem.mean().item(), b.mean().item(), alpha.mean().item(),ro.mean().item())
        if return_state:
            return torch.stack(ty), (mem, y, b)
        elif self.return_mem:
            return mem.unsqueeze(0)
        else:
            return torch.stack(ty)

class DHSNN(BaseNeuron):
    """
        Altered from https://github.com/eva1801/DH-SNN
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            input_features: int = 1,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,
            exec_mode: str = "serial",
            recurrent: bool = False,
            branch: int = 4
    ):
        super(DHSNN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.input_features = input_features
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent

        if self.recurrent:
            self.pad = ((input_features + neuron_num) // branch * branch + branch - (
                        input_features + neuron_num)) % branch
            self.dense = nn.Linear(input_features + neuron_num + self.pad, neuron_num * branch)
            # bound = 1 / math.sqrt(input_features) + 1 / math.sqrt(neuron_num)
            # nn.init.uniform_(self.dense.bias, -bound, bound)


        else:
            self.pad = ((input_features) // branch * branch + branch - (input_features)) % branch
            self.dense = nn.Linear(input_features + self.pad, neuron_num * branch)

        # mask_rate = 1 / branch

        self.tau_m = nn.Parameter(torch.Tensor(self.neuron_num))
        self.tau_n = nn.Parameter(torch.Tensor(self.neuron_num, branch))
        # the number of dendritic branch
        self.branch = branch
        self.create_mask()
        # if self.recurrent:
        #     nn.init.uniform_(self.tau_m, 0, 0)
        #     nn.init.uniform_(self.tau_n, 4, 6)
        # else:
        nn.init.uniform_(self.tau_m, 4, 6)
        nn.init.uniform_(self.tau_n, 0, 4)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}, "
            f"branch={self.branch}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            d_input = state[2]
            return_state = True
        else:
            v = torch.ones(tx.size(1), self.neuron_num, device=tx.device) * self.rest
            y = torch.zeros(tx.size(1), self.neuron_num, device=tx.device)
            d_input = torch.zeros(tx.size(1), self.neuron_num, self.branch, device=tx.device)
            return_state = False

        for x in tx:
            beta = torch.sigmoid(self.tau_n)
            padding = torch.zeros(x.size(0), self.pad).to(x.device)
            if self.recurrent:
                x = torch.cat((x.float(), y, padding), 1)
            else:
                x = torch.cat((x.float(), padding), 1)
            x = self.dense(x)

            # update dendritic currents

            d_input = beta * d_input + (1 - beta) * x.reshape(-1, self.neuron_num, self.branch)
            # summation of dendritic currents

            l_input = d_input.sum(dim=2, keepdim=False)

            alpha = torch.sigmoid(self.tau_m)

            v = v * alpha + l_input - self.threshold * y  # replace orginal to avoid gradient vanishing
            # v = v * alpha + (1 - alpha) * l_input - self.threshold * y
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)

        # print('tv',torch.stack(tv).mean())
        if return_state:
            return torch.stack(ty), (v, y, d_input)
        else:
            return torch.stack(ty)

    def create_mask(self):
        if self.recurrent:
            input_size = self.input_features + self.neuron_num + self.pad
        else:
            input_size = self.input_features + self.pad
        self.mask = torch.zeros(self.neuron_num * self.branch, input_size).cuda()
        for i in range(self.neuron_num):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[
                    i * self.branch + j, seq[j * input_size // self.branch:(j + 1) * input_size // self.branch]] = 1

    def apply_mask(self):
        # print('Apply Mask')
        # if self.recurrent:
        #     self.dense.weight.data = self.dense.weight.data * self.mask[:,:(self.input_features + self.pad)]
        #
        #     self.recurrent_weight.weight.data = self.recurrent_weight.weight.data * self.mask[:,(self.input_features + self.pad):]
        # else:

        self.dense.weight.data = self.dense.weight.data * self.mask
    # def _temporal_fused_process(self, tx):
    # else: # todo: add recurrent acceleration


class adLIF(BaseNeuron):
    """
        Altered from https://github.com/idiap/sparch
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(adLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent

        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
            nn.init.orthogonal_(self.recurrent_weight.weight)

        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        # self.a_lim = [-1.0, 1.0]
        # self.b_lim = [0.0, 2.0] # the original setting is hard to converge

        self.a_lim = [0.0, 1.0]
        self.b_lim = [0.0, 1.0]

        # Trainable parameters
        self.alpha = nn.Parameter(torch.Tensor(self.neuron_num))
        self.beta = nn.Parameter(torch.Tensor(self.neuron_num))
        self.a = nn.Parameter(torch.Tensor(self.neuron_num))
        self.b = nn.Parameter(torch.Tensor(self.neuron_num))

        self.norm = nn.BatchNorm1d(self.neuron_num, momentum=0.05)

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            wt = state[2]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            wt = torch.zeros_like(tx[0])
            return_state = False

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        _tx = self.norm(tx.reshape(tx.shape[0] * tx.shape[1], tx.shape[2]))
        tx = _tx.reshape(tx.shape[0], tx.shape[1], tx.shape[2])

        for x in tx:
            if self.recurrent:
                # Set diagonal elements of recurrent matrix to zero
                r_weight = self.recurrent_weight.weight.clone().fill_diagonal_(0)
                x = x + torch.matmul(y, r_weight)
            # Compute potential (adLIF)
            wt = beta * wt + a * v + b * y
            v = alpha * (v - y) + (1 - alpha) * (x - wt)

            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)

        if return_state:
            return torch.stack(ty), (v, y, wt)
        else:
            return torch.stack(ty)
    # def _temporal_fused_process(self, tx):
    # else: # todo: add recurrent acceleration

class PMSN_surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, gamma=1.):
        # tm=torch.arange(input.size(-1),device=input.device).repeat(input.size(0),input.size(1),1) * thresh + (2-1e-3) * thresh
        cum_x = input.cumsum(dim=-1)
        cum_x_shift = cum_x.clone()
        cum_x_shift[..., 1:] = cum_x[..., :-1]
        cum_x_shift[..., 0] = 0
        spike_shift = (cum_x_shift / thresh).floor().clamp(min=0)
        out = ((cum_x - spike_shift * thresh) / thresh).floor().clamp(min=0, max=1)
        L = torch.tensor([gamma])
        ctx.save_for_backward(thresh, cum_x - spike_shift * thresh, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (thresh, delta, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        # tmp = (1 / gamma) * (1 / gamma) * ((gamma - abs(delta-thresh)).clamp(min=0))  # triangle
        tmp = (gamma - abs(delta - thresh) > 0) * gamma  # rectangle
        grad_output = grad_input * tmp
        return grad_output, None


class PMSN(BaseNeuron):

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  # TODO: Set a default value
            exec_mode: str = "serial",
            recurrent: bool = False,
    ):
        super(PMSN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.return_mem = False

        self.kernel = PMSN_kernel(self.neuron_num, N=4)
        self.D = nn.Parameter(torch.randn(self.neuron_num))
        self.thresh = torch.tensor([self.threshold])
        self.bn = nn.BatchNorm1d(self.neuron_num)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            # f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        if isinstance(state, tuple):
            tx[0,] = tx[0,] + state[0]
            return_state = True
        else:
            return_state = False
        step_num = tx.size(0)  # (T,B,H)

        tx = self.bn(tx.view(-1, tx.size(-1))).view(step_num, -1, self.neuron_num)
        tx = tx.permute(1, 2, 0)  # (B, H, T)
        # Compute SSM Kernel
        k = self.kernel(L=step_num, u=tx)  # (H T)

        # # Convolution
        k_f = torch.fft.rfft(k, n=2 * step_num)  # (H T)
        u_f = torch.fft.rfft(tx, n=2 * step_num)  # (B H T)
        _y = torch.fft.irfft(u_f * k_f, n=2 * step_num)[..., :step_num]  # (B H T)
        y = _y + (tx * self.D.unsqueeze(-1))
        # proposed reset mechanism
        ty = PMSN_surrogate.apply(y.relu(), self.thresh.to(tx.device))
        ty = ty.permute(2, 0, 1)

        if return_state:
            return ty, (_y[..., -1], None)
        elif self.return_mem:
            return y[-1,].unsqueeze(0)
        else:
            return ty


class PMSN_kernel(nn.Module):
    def __init__(self, d_model, N=4, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H).uniform_(0, 1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)  # [H]

        self.log_dt = nn.Parameter(log_dt)
        diag_indices = torch.arange(N)
        sub_diag_indices = diag_indices[:-1] + 1
        super_diag_indices = diag_indices[1:] - 1

        S = torch.zeros(N, N)
        S[diag_indices, diag_indices] = -0.5
        S[diag_indices[:-1], sub_diag_indices] = 5. * ((torch.arange(N - 1) + 1))
        S[diag_indices[1:], super_diag_indices] = -5. * ((torch.arange(N - 1) + 1))  # 超对角线

        S_diag = torch.diagonal(S)
        A_real = (torch.mean(S_diag) * torch.ones_like(S_diag)).unsqueeze(0).repeat(H, 1)

        A_imag, V = torch.linalg.eigh(S * -1j)  # [N; N,N]
        A_imag = A_imag.unsqueeze(0).repeat(H, 1)

        self.mask = torch.zeros(N, N).cuda()
        self.mask[diag_indices, diag_indices] = 1
        self.mask[diag_indices[:-1], sub_diag_indices] = 1

        log_A_real = torch.log(-A_real)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        B = torch.ones(H, N)
        C = torch.zeros(H, N)
        C[:, -1] = 1
        Vinv = V.conj().T  # [N,N]
        CV = torch.einsum('hm,mn->hn', C + 0j, V)  # [H,N]
        VinvB = torch.einsum('mn,hn->hm', Vinv, B + 0j)  # [H,N]

        self.VinvB_real = nn.Parameter(VinvB.real)
        self.VinvB_imag = nn.Parameter(VinvB.imag)
        self.CV_real = nn.Parameter(CV.real)
        self.CV_imag = nn.Parameter(CV.imag)

    def forward(self, L, u=None):
        # u [B,H,L]
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)
        B = self.VinvB_real + 1j * self.VinvB_imag  # (H,N)
        C = self.CV_real + self.CV_imag * 1j

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H,1)
        A_bar = torch.exp(A * dt.unsqueeze(-1))  # [H N]
        B_bar = (A_bar - 1) * B / A
        # Vandermonde multiplication
        logK = (A * dt.unsqueeze(-1)).unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)   e-At
        K = torch.exp(logK)
        KB = torch.einsum('hnl,hn->hnl', K, B_bar)  # e-At*B  # (H N L)
        CKB = torch.einsum('hn, hnl -> hl', C, KB).real  # (H L)
        return CKB