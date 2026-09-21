import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    a,b,c,h = float(a), float(b), float(c), float(h)

    def func(a,b,c):
        return (a*b + c)

    d = func(a,b,c)

    par_a = (func((a+h), b, c) - d)/h
    par_b = (func(a, (b+h), c) - d)/h
    par_c = (func(a, b, (c+h)) - d)/h

    return (d, par_a, par_b, par_c)
