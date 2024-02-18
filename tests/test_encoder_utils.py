import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytest
import torch
from einops import rearrange, repeat
from PIL import Image

from encoders.encoder import BaseModel

img_path = Path("tests/med_res.jpg")
save_path = Path(__file__).parent / "output"


def get_img(
    img_type: Union[np.ndarray, Image.Image, torch.Tensor],
    hwc_order=False,
    dtype=None,
    device=None,
    batched: Optional[int] = 1,
    image_shape: Optional[tuple[int]] = None,
):

    img = Image.open(img_path)

    if img_type == Image.Image:
        return img

    img = np.array(img)
    if img_type == torch.Tensor:
        img = torch.from_numpy(img)

    if image_shape is not None:
        img = img[: image_shape[0], : image_shape[1]]

    if not hwc_order:
        img = rearrange(img, "h w c -> c h w")

    if dtype is not None:
        img = img / 255.0
        if img_type == torch.Tensor:
            img = img.to(dtype=dtype)
        elif img_type == np.ndarray:
            img = img.astype(dtype)

    if device is not None and img_type == torch.Tensor:
        img = img.to(device=device)

    if batched is not None:
        img = repeat(img, f"... -> b ...", b=batched)

    return img


valid_configs = [
    {"img_type": Image.Image},
    {"img_type": np.ndarray, "hwc_order": True, "batched": None},
    {"img_type": torch.Tensor, "dtype": torch.float32, "batched": None},
    {"img_type": torch.Tensor, "hwc_order": False, "dtype": torch.bfloat16},
    {"img_type": torch.Tensor, "image_shape": (224, 224), "dtype": torch.bfloat16},
]


@pytest.mark.parametrize("model_data", [(model_name, model_cls) for model_name, model_cls in BaseModel._registry.items()])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
def test_encoders(model_data, model_dtype):
    device = torch.device("cuda:0")
    model_name, model_cls = model_data
    model = model_cls(dtype=model_dtype, device=device)
    print(f"Running {model_name}...")
    for img_params in valid_configs:
        if model_name == "ViTFeatureExtractor" and "image_shape" not in img_params:
            continue
        if model_name == "ResNetFeatureExtractor":
            continue
        image = get_img(**img_params)
        torch.cuda.synchronize()
        start = time.time()
        output = model(image)
        end = time.time()
        torch.cuda.synchronize()
        output_shape = output.shape if isinstance(output, torch.Tensor) else [(k, v.shape) for k, v in output.items()]
        print(f"{model_name}: {(end - start) * 1000}ms, {output_shape}")
