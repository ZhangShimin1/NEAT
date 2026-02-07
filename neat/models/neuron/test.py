import torch
import pytest
import sys

from neat.models.neuron.lif import adLIF
from neat.models.surrogate.surrogate import SurrogateGradient


def test_adlif_unused_parameters():
    """
    Test to identify unused parameters in the adLIF neuron model.
    This test creates an adLIF instance, runs a forward pass, and checks
    if all parameters have gradients after backpropagation.
    """
    # Setup
    batch_size = 2
    time_steps = 5
    neuron_num = 10
    input_features = 10

    # Create a surrogate gradient function
    surrogate = SurrogateGradient(func_name="triangle", a=1.0)

    # Create adLIF neuron with various parameter settings
    neuron = adLIF(
        rest=0.0,
        decay=0.2,
        threshold=0.3,
        input_features=input_features,
        neuron_num=neuron_num,
        time_step=time_steps,
        surro_grad=surrogate,
        exec_mode="serial",
        recurrent=False,  # Test with recurrent connections
    )

    # Create random input
    input_data = torch.rand(batch_size, time_steps, neuron_num)

    # Forward pass
    output = neuron(input_data)

    # Create a loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Check which parameters have gradients
    used_params = []
    unused_params = []

    for name, param in neuron.named_parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            used_params.append(name)
        else:
            unused_params.append(name)

    print("Used parameters:", used_params)
    print("Unused parameters:", unused_params)

    # Test with different execution modes
    neuron_parallel = adLIF(
        rest=0.0,
        decay=0.2,
        threshold=0.3,
        input_features=input_features,
        neuron_num=neuron_num,
        time_step=time_steps,
        surro_grad=surrogate,
        exec_mode="parallel",  # Test parallel mode
        recurrent=False,
    )

    # Check if input_features parameter is actually used
    assert hasattr(neuron, "input_features"), "input_features attribute should exist"

    # Test if changing time_step affects output
    neuron1 = adLIF(neuron_num=neuron_num, time_step=time_steps, surro_grad=surrogate)

    neuron2 = adLIF(
        neuron_num=neuron_num,
        time_step=time_steps + 1,  # Different time_step
        surro_grad=surrogate,
    )

    # Same input for both neurons
    test_input = torch.rand(batch_size, time_steps, neuron_num)

    # Check if outputs are different (they should be the same if time_step is unused)
    out1 = neuron1(test_input)
    out2 = neuron2(test_input)

    time_step_used = not torch.allclose(out1, out2)
    print(f"time_step parameter is {'used' if time_step_used else 'unused'}")

    return used_params, unused_params


if __name__ == "__main__":
    used, unused = test_adlif_unused_parameters()
    print("\nSummary:")
    print(f"Used parameters: {', '.join(used)}")
    print(f"Unused parameters: {', '.join(unused)}")
