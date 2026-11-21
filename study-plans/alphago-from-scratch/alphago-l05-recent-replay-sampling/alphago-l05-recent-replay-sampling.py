import copy

def sample_recent_replay_batch(
    games: list | tuple,
    window_size: int,
    position_indices: list | tuple,
) -> tuple[list, int]:
    """
    Returns independent sampled records and the eligible position count.
    """
    recent_games = games[-window_size:] if len(games) > 0 else []

    eligible_pool = []
    for game in recent_games:
        eligible_pool.extend(game)

    eligible_count = len(eligible_pool)

    selected_records = [copy.deepcopy(eligible_pool[i]) for i in position_indices]

    return selected_records, eligible_count