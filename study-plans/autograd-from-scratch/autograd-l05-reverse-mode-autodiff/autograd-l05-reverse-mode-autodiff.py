import numpy as np
import heapq

def reverse_mode_autodiff(leaves, operations, output_id):
    """
    Returns: `(output_value, gradients)` with one gradient for every reachable leaf ID.
    """
    
    leaf_lookup = {l['id']: l for l in leaves}
    op_lookup = {op['id']: op for op in operations}
    
    all_ids = [l['id'] for l in leaves] + [op['id'] for op in operations]
    position = {nid: i for i,nid in enumerate(all_ids)}
    
    # Step-2
    def parents_of(nid):
      if nid in op_lookup:
        return op_lookup[nid]['parents']
      return []
    
    
    visited = set()
    stack = [output_id]
    
    while stack:
      nid = stack.pop()
      if nid in visited: continue
      visited.add(nid)
    
      for p in parents_of(nid):
        stack.append(p)
    
    
    in_deg = {nid: 0 for nid in visited}
    children = {nid: [] for nid in visited} 
    
    for nid in all_ids:
      if nid not in visited: continue
      for p in parents_of(nid):
        in_deg[nid] += 1
        children[p].append(nid)
    
    ready = [(position[nid], nid) for nid in visited if in_deg[nid]==0]
    heapq.heapify(ready)
    
    topo_order = []
    while ready:
      _, nid = heapq.heappop(ready)
      topo_order.append(nid)
      for child in children[nid]:
        in_deg[child] -= 1
        if in_deg[child] == 0:
          heapq.heappush(ready, (position[child], child))
    
    values = {}
    for nid in topo_order:
      if nid in leaf_lookup:
        values[nid] = float(leaf_lookup[nid]['value'])
      else:
        op = op_lookup[nid]
        p = op['parents']
        if op['op'] == 'add':
          values[nid] = values[p[0]] + values[p[1]]
        elif op['op'] == 'mul':
          values[nid] = values[p[0]] * values[p[1]]
        elif op['op'] == 'tanh':
          values[nid] = float(np.tanh(values[p[0]]))
    
    grads = {nid: 0.0 for nid in visited}
    grads[output_id] = 1.0
    
    for nid in reversed(topo_order):
      g = grads[nid]
      if nid in leaf_lookup: continue
      op = op_lookup[nid]
      p = op['parents']
    
      if op['op'] == 'add':
        grads[p[0]] += g
        grads[p[1]] += g
      elif op['op'] == 'mul':
        grads[p[0]] += g * values[p[1]]
        grads[p[1]] += g * values[p[0]]
      elif op['op'] == 'tanh':
        grads[p[0]] += g * (1 - values[nid]**2)
    
    leaf_grads = {}
    for leaf in leaves:
      if leaf['id'] in visited:
        leaf_grads[leaf['id']] = float(grads[leaf['id']])
    
    return (float(values[output_id]), leaf_grads)
