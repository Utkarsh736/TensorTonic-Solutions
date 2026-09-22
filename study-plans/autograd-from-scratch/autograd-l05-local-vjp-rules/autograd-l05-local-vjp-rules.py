import numpy as np

def local_vjp(operation, inputs, output, upstream_gradient):
    """
    Returns: one gradient contribution per input in input order.
    """
    g = upstream_gradient
    
    if operation=="add": return [g, g]
    elif operation=="mul": return [g*inputs[1], g*inputs[0]]
    else: return [g*(1-(output**2))]