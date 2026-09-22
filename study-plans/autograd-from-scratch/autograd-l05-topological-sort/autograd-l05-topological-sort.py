import numpy as np
import heapq

def topological_sort(nodes, output_id):
    """
    Returns: reachable node IDs in deterministic topological order using input order to break valid ordering ties.
    """
    lookup = {node['id']: node for node in nodes}
    position = {node['id']: i for i, node in enumerate(nodes)}

    reachable = set()
    stack = [output_id]

    while stack:
      nid = stack.pop()
      if nid in reachable: continue
      reachable.add(nid)
      for p in lookup[nid]['parents']:
        stack.append(p)

    in_degree = {nid: 0 for nid in reachable}
    children = {nid: [] for nid in reachable}

    for n in nodes:
      if n['id'] not in reachable: continue
      for p in n['parents']:
        in_degree[n['id']] += 1
        children[p].append(n['id'])


    ready = [(position[nid], nid) for nid in reachable if in_degree[nid] == 0]
    heapq.heapify(ready)

    order = []
    while ready:
        pos, nid = heapq.heappop(ready)
        order.append(nid)
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready, (position[child], child))

    return order