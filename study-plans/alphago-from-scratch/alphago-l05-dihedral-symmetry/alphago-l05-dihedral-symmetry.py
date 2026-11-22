import numpy as np

def apply_go_symmetry(
    state_planes: np.ndarray,
    policy: np.ndarray,
    value: float,
    transform_id: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns transformed state and policy arrays, followed by the unchanged value.
    """
    state = np.asarray(state_planes)
    pol = np.asarray(policy)

    n = state.shape[-1]

    def transform(arr):
        if transform_id < 4:
            k = transform_id
            out = np.rot90(arr, k=k, axes=(-2, -1))
        else:
            k = transform_id - 4
            out = np.flip(arr, axis=-1)
            out = np.rot90(out, k=k, axes=(-2, -1))

        return np.array(out, copy=True, dtype=arr.dtype)

    transformed_state = transform(state)

    board_policy = pol[:-1].reshape(n, n)
    transformed_board = transform(board_policy)

    transformed_policy = np.empty_like(pol)
    transformed_policy[:-1] = transformed_board.reshape(-1)
    transformed_policy[-1] = pol[-1]

    return transformed_state, transformed_policy, value
