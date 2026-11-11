import torch

def encode_go_history(history: list[torch.Tensor], to_play: int) -> torch.Tensor:
    """
    Returns: a float32 tensor of shape (17, board size, board size) on the history device.
    """
    device = history[0].device
    h, w = history[0].shape

    planes = []

    for i in range(8):
        if i < len(history):
            board = history[i]

            current = (board == to_play).to(torch.float32)

            opponent = (board == -to_play).to(torch.float32)
        else:
            current = torch.zeros((h, w), dtype=torch.float32, device=device)
            opponent = torch.zeros((h, w), dtype=torch.float32, device=device)

        planes.append(current)
        planes.append(opponent)

    if to_play == 1:
        colour_plane = torch.ones((h, w), dtype=torch.float32, device=device)
    else:
        colour_plane = torch.zeros((h, w), dtype=torch.float32, device=device)

    planes.append(colour_plane)

    return torch.stack(planes, dim=0)
