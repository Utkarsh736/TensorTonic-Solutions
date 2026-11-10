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

def resolve_go_action(board: list, player: int, action: int, consecutive_passes: int, move_count: int) -> tuple:
    """
    Returns: a tuple: next board, player, captures, consecutive passes, move count, terminal.
    """
    board_np = np.array(board, dtype=int)
    n = board_np.shape[0]
    board_area = n * n

    captured_count = 0
    captured_stones = set()

    if action == board_area:
        consecutive_passes += 1
    else:
        # Placement
        r, c = divmod(action, n)
        board_np[r, c] = player
        consecutive_passes = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        groups_to_capture = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and board_np[nr, nc] == -player:
                opp_group, opp_libs = get_group_and_liberties(board_np, nr, nc)
                if not opp_libs:
                    groups_to_capture.append(opp_group)

        for group in groups_to_capture:
            for gr, gc in group:
                if (gr, gc) not in captured_stones:
                    captured_stones.add((gr, gc))
                    board_np[gr, gc] = 0
                    captured_count += 1

    next_player = -player
    move_count += 1

    terminal = (consecutive_passes >= 2) or (move_count >= 2 * board_area)

    return (board_np, next_player, captured_count,
            consecutive_passes, move_count, terminal)
