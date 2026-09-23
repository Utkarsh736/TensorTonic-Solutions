import numpy as np
import heapq

def autodiff_gradient_check(leaves: list, operations: list, output_id: str, h: float) -> tuple:
    """
    Returns a tuple of analytic gradients, numerical gradients, and maximum error.
    """
    leaf_lookup = {l['id']: l for l in leaves}
    op_lookup = {op['id']: op for op in operations}
    
    all_ids = [l['id'] for l in leaves] + [op['id'] for op in operations]
    position = {nid:i for i,nid in enumerate(all_ids)}
    
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
    
    in_deg = {nid:0 for nid in visited}
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
    
    
    def forward_output(leaf_values):
        vals = {}
        for nid in topo_order:
            if nid in leaf_lookup:
                vals[nid] = leaf_values[nid]
            else:
                op = op_lookup[nid]
                p = op['parents']
                if op['op'] == 'add':
                    vals[nid] = vals[p[0]] + vals[p[1]]
                elif op['op'] == 'mul':
                    vals[nid] = vals[p[0]] * vals[p[1]]
                elif op['op'] == 'tanh':
                    vals[nid] = float(np.tanh(vals[p[0]]))
        return vals
    
    
    base_leaf_values = {l['id']: float(l['value']) for l in leaves}
    values = forward_output(base_leaf_values)
    baseline_output = values[output_id]
    
    
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
    
    
    numerical = {}
    for leaf in leaves:
        if leaf['id'] not in visited:
            continue
        perturbed = dict(base_leaf_values)
        perturbed[leaf['id']] += h
        y_pert = forward_output(perturbed)[output_id]
        numerical[leaf['id']] = float((y_pert - baseline_output) / h)
    
    analytic = {}
    for leaf in leaves:
        if leaf['id'] in visited:
            analytic[leaf['id']] = float(grads[leaf['id']])
    
    max_err = 0.0
    for k in analytic:
        diff = abs(analytic[k] - numerical[k])
        if diff > max_err:
            max_err = diff
    max_err = float(max_err)

    return (analytic, numerical, max_err)