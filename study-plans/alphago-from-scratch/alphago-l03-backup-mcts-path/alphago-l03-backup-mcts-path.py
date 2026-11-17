def backup_mcts_path(path: list, leaf_value: float, leaf_player: int) -> list:
    """
    Returns updated copies of the path records in their original order.
    """
    updated_path = copy.deepcopy(path)

    for record in updated_path:
        # Value from this edge player's perspective.
        if record["player"] == leaf_player:
            v = leaf_value
        else:
            v = -leaf_value

        old_n = record["N"]
        old_w = record["W"]

        new_n = old_n + 1
        new_w = old_w + v
        new_q = new_w / new_n

        record["N"] = new_n
        record["W"] = new_w
        record["Q"] = new_q

    return updated_path
