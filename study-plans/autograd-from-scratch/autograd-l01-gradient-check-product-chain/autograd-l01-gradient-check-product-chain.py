import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    val_list = [a,b,c,f,h]
    a,b,c,f,h = [float(i) for i in val_list]

    def func_l(a,b,c,f): return (a*b+c)*f
    
    e = a*b + c
    l = e*f

    an_a = f*b
    an_b = f*a
    an_c = f
    an_f = e

    an_list = [an_a, an_b, an_c, an_f]
    
    nu_a = (func_l(a+h, b, c, f)-l)/h
    nu_b = (func_l(a, b+h, c, f)-l)/h
    nu_c = (func_l(a, b, c+h, f)-l)/h
    nu_f = (func_l(a, b, c, f+h)-l)/h

    nu_list = [nu_a, nu_b, nu_c, nu_f]

    max_dis = max(abs(a-n) for a,n in zip(an_list, nu_list))

    return (l, an_list, nu_list, max_dis)
    
