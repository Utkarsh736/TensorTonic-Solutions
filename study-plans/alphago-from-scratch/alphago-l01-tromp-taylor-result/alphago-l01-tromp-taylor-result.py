import numpy as np

def tromp_taylor_result(board: list, komi: float, consecutive_passes: int, move_count: int) -> tuple:
    """
    Returns: a tuple: terminal, Black score, White score, winner; unfinished scores and winner are None.
    """
    board_np = np.array(board, dtype=int)
    n = board_np.shape[0]
    board_area = n * n

    if consecutive_passes < 2 and move_count < 2 * board_area:
        return (False, None, None, None)

    black_stones = int(np.count_nonzero(board_np == 1))
    white_stones = int(np.count_nonzero(board_np == -1))

    visited = np.zeros((n, n), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    black_territory = 0
    white_territory = 0

    for r0 in range(n):
        for c0 in range(n):
            if board_np[r0, c0] != 0 or visited[r0, c0]:
                continue

            region_cells = []
            border_colors = set()
            stack = [(r0, c0)]

            while stack:
                r, c = stack.pop()
                if visited[r, c]:
                    continue
                visited[r, c] = True
                region_cells.append((r, c))

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        val = board_np[nr, nc]
                        if val == 0:
                            # still inside the empty region
                            if not visited[nr, nc]:
                                stack.append((nr, nc))
                        else:
                            # stone adjacent to the region → record its colour
                            border_colors.add(int(val))

            size = len(region_cells)
            if border_colors == {1}:
                black_territory += size
            elif border_colors == {-1}:
                white_territory += size

    black_score = float(black_stones + black_territory)
    white_score = float(white_stones + white_territory + komi)

    if black_score > white_score:
        winner = 1
    elif white_score > black_score:
        winner = -1
    else:
        winner = 0

    return (True, black_score, white_score, winner)
