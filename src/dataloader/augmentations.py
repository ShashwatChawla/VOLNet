def mask_data(data: torch.Tensor, min_mask: float = 0.25, max_mask: float = 0.75) -> torch.Tensor:
    """
    Mask the data with random values between min_mask and max_mask.
    """
    B, N, C, H, W = data.shape