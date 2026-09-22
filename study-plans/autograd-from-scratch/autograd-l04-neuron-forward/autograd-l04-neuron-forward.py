import torch

def neuron_forward(inputs, weights, bias):
    """
    Returns: scalar preactivation and tanh output
    """
    a = torch.sum(inputs*weights) + bias

    y = torch.tanh(a)

    return (a, y)
