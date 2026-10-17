import numpy as np

def exp_power_backward(x: float, exponent: float, upstream_gradient: float) -> tuple:
    """
    Returns a tuple of exp output, exp gradient, power output, and power gradient.
    """
    x = float(x)
    exponent = float(exponent)
    upstream_gradient = float(upstream_gradient)
    
    y_exp = float(np.exp(x))
    g_exp = upstream_gradient*y_exp

    if exponent == 0:
        y_pow = 1.0
        g_pow = 0.0

    else:
        y_pow = float(x**exponent)
        local_derivative = exponent * (x ** (exponent - 1))
        g_pow = float(upstream_gradient * local_derivative)

    return (y_exp, g_exp, y_pow, g_pow)
