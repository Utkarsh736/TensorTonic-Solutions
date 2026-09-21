import numpy as np

def build_expression_graph(leaves, operations):
    """
    Returns: node records in creation order and the final node ID
    """
    registry = {}
    nodes = []

    for leaf in leaves:
        node = {
            'id': leaf['id'],
            'data': float(leaf['data']),
            'grad': 0.0,
            'op':'',
            'parents': [],
        }

        nodes.append(node)
        
        registry[leaf['id']] = node

    for op in operations:
        l_val = registry[op['left']]['data']
        r_val = registry[op['right']]['data']

        if op['op'] == "+": data = l_val+r_val
        elif op['op'] == "*": data = l_val*r_val

        node = {
           'id': op['id'],
           'data': data,
           'grad': 0.0,
           'op': op['op'],
           'parents': [op['left'], op['right']],
       }

        nodes.append(node)
        registry[op['id']] = node


    final_id = nodes[-1]['id']
    return (nodes, final_id)

        
