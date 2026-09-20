import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    coeffs = np.array(coefficients, dtype=np.float64)
    x = np.float64(x)
    h = np.float64(h)

    def eval(pt):
        acc = np.float64(0.0)
        for c in reversed(coeffs):
            acc = acc * pt + c
        return acc

    y_x = eval(x)
    y_xh = eval(x + h)

    slope = (y_xh - y_x) / h

    return (float(y_x), float(y_xh), float(slope))