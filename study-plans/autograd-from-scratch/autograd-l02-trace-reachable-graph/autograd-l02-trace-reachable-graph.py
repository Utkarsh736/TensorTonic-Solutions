import numpy as np

def trace_reachable_graph(nodes, output_id):
    """
    Returns: reachable node IDs and parent-to-child edges in deterministic order
    """
    lookup = {node['id']: node for node in nodes}

    visited = set()
    stack = [output_id]

    while stack:
      nid = stack.pop()
      if nid in visited: continue
      visited.add(nid)
      for p in lookup[nid]['parents']:
        stack.append(p)

    ids = [n['id'] for n in nodes if n['id'] in visited]

    edges = []
    for n in nodes:
      if n['id'] not in visited: continue
      for p in n['parents']:
        edges.append([p, n['id']])

    return(ids, edges)