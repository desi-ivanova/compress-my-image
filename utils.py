from typing import Sequence
import numpy as np
from PIL import Image

import rawpy
import torch


def load_image(file):
    """Load a png, jpeg or dng image

    Args:
        file: path to the image

    Returns:
        unsigned 8-bit (uint8) integer np.array
    """
    if file.lower().endswith(".dng"):
        with rawpy.imread(file) as raw:
            return raw.postprocess()  # np array of shape [H, W, C]
    else:
        return np.array(Image.open(file, formats=["png", "jpeg"]))


def rgb_to_tensor(rgb_img: np.array) -> torch.Tensor:
    """Convert an rgb np array to float tensor

    Args:
        rgb_img: np array of shape [H, W, C]

    Returns:
        torch tensor of shape [1, 3, H, W]
    """
    assert (
        rgb_img.shape[-1] == 3 and rgb_img.ndim == 3
    ), f"Expected [H, W, 3], got {rgb_img.shape}"

    output = torch.tensor(rgb_img, dtype=torch.float32) / 255.0
    output = output.permute(-1, -3, -2)  # [H, W, 3] -> [3, H, W]
    return output.unsqueeze(0)  # [1, 3, H, W]


def tensor_to_rgb(tensor_img: torch.Tensor) -> np.array:
    """Convert a tensor to rgb np array

    Args:
        tensor_img: tensor of shape [1, 3, H, W]

    Returns:
        unsigned 8-bit (uint8) integer np.array of shape [H, W, 3]
    """
    output = tensor_img.squeeze(0).permute(-2, -1, 0)
    output = torch.clamp(output, min=0.0, max=1.0)
    return (output * 255.0 + 0.5).type(torch.uint8).numpy()


def pad_image(img: torch.Tensor, sizes: Sequence[int], factor: int):
    """Pad an image so its dimensions are multiples of factor

    Args:
        img: torch tensor of shape [..., H, W]
        sizes: height and width of the image (img.shape[-2:])
        factor: factor that the padded image should be multiple of

    Returns:
        padded torch tensor
    """
    pad_h, pad_w = [(factor - (s % factor)) % factor for s in sizes]
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), "reflect")
