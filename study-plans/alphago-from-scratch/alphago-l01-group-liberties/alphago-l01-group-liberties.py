import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    board_np = np.array(board)
    n = board_np.shape[0]
    
    color = board_np[row, col]
    
    if color == 0:
        return ([], [])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    visited = set()        # Stones in the group
    liberties = set()     
    stack = [(row, col)]
    
    # Iterative DFS
    while stack:
        r, c = stack.pop()
        
        if (r, c) in visited:
            continue
        
        visited.add((r, c))
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n and 0 <= nc < n:
                neighbour_val = board_np[nr, nc]
                
                if neighbour_val == 0:
                    liberties.add((nr, nc))
                elif neighbour_val == color:
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))

    group_list = [[r, c] for r, c in sorted(visited)]
    liberty_list = [[r, c] for r, c in sorted(liberties)]
    
    return (group_list, liberty_list)
