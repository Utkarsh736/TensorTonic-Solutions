import numpy as np

def value_multiplication_node(left, right, output_id):
    """
    Returns: a multiplication node that retains the two supplied leaf records as ordered parents
    """
    data_l = float(left['data'])
    data_r = float(right['data'])

    out = {'id': output_id,
           'data': data_l*data_r,
           'grad': 0.0,
           'op': '*',
           'parents': [left, right],
          }

    return out
