import numpy as np

def get_group_and_liberties(board_np, row, col):
    n = board_np.shape[0]
    color = board_np[row, col]
    if color == 0:
        return set(), set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    group = set()
    liberties = set()
    stack = [(row, col)]

    while stack:
        r, c = stack.pop()
        if (r, c) in group:
            continue
        group.add((r, c))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                val = board_np[nr, nc]
                if val == 0:
                    liberties.add((nr, nc))
                elif val == color and (nr, nc) not in group:
                    stack.append((nr, nc))

    return group, liberties

def legal_go_action_mask(board: list, player: int, seen_positions: list, consecutive_passes: int, move_count: int) -> np.ndarray:
    """
    Returns: a Boolean NumPy array of legality flags, with pass last.
    """
    board_np = np.array(board, dtype=int)
    n = board_np.shape[0]
    board_area = n * n

    if consecutive_passes >= 2 or move_count >= 2 * board_area:
        return np.zeros(board_area + 1, dtype=bool)

    seen_boards = [np.array(sp, dtype=int) for sp in seen_positions]

    mask = np.zeros(board_area + 1, dtype=bool)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for action in range(board_area):
        r, c = divmod(action, n)

        if board_np[r, c] != 0:
            continue

        cand = board_np.copy()
        cand[r, c] = player

        captured = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if cand[nr, nc] == -player:
                    opp_group, opp_libs = get_group_and_liberties(cand, nr, nc)
                    if not opp_libs:
                        captured.extend(opp_group)

        for cr, cc in captured:
            cand[cr, cc] = 0

        own_group, own_libs = get_group_and_liberties(cand, r, c)
        if not own_libs:
            continue

        if any(np.array_equal(cand, sb) for sb in seen_boards):
            continue

        # All checks passed → intersection action is legal
        mask[action] = True

    mask[board_area] = True

    return mask
