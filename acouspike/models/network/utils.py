def reset_states(model):
    for name, m in model.named_modules():
        if hasattr(m, 'reset'):
            # print(f'before {name}:  {m.v}')
            # if not isinstance(m, MemoryModule):
            #     print(f'Trying to call `reset()` of {m}, which is not base.MemoryModule')
            m.reset()

def count_parameters(net):
    ''' Count number of parameters in model influenced by global loss. '''
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

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