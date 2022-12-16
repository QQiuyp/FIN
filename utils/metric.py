from utils.utils import *
from torch.nn.functional import mse_loss as mse

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    input = clamp(((input.detach().cpu().squeeze()/2)+0.5) * max_val, 0, max_val)
    target = clamp(((target.detach().cpu().squeeze()/2)+0.5) * max_val, 0, max_val)
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    return 10.0 * torch.log10(max_val ** 2 / mse(input, target, reduction='mean'))

