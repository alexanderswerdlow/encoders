from contextlib import nullcontext
import functools
import math
from abc import ABC, abstractmethod, abstractproperty
from functools import partial
from pyexpat import features
from typing import Callable, Optional, TypeAlias, Union

import imageio.v3 as iio
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import Tensor
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)

from encoders.extracted_encoder_utils import (interpolate_embeddings,
                                              pad_image_and_adjust_coords)

import inspect
ImArr: TypeAlias = Union[Image.Image, Tensor, np.ndarray]


def reshape_vit_output(num_to_truncate: int, x: torch.Tensor):
    x = x[:, num_to_truncate:]
    assert np.isclose(int(math.sqrt(x.shape[1])), math.sqrt(x.shape[1]))
    h, w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
    x = rearrange(x, "b (h w) d -> b d h w", h=h, w=w)
    return {"x": x}


def identity(**kwargs):
    return kwargs


def get_feature_layer(**kwargs):
    kwargs["x"] = kwargs["x"][-(kwargs["num_from_back"] + 1)]
    return kwargs


def swin_rearrange(**kwargs):
    kwargs = get_feature_layer(**kwargs)
    kwargs["x"] = rearrange(kwargs["x"], "b h w d -> b d h w")
    return kwargs


class BaseModel(ABC, nn.Module):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            init_params = inspect.signature(cls.__init__).parameters.values()
            if all(param.default is not param.empty for param in init_params 
                   if param.name != 'self' and param.kind != param.VAR_KEYWORD):
                BaseModel._registry[cls.__name__] = cls

    def __init__(
        self,
        compile: bool = False,
        compile_kwargs: dict = {},
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        deferred_init: bool = False,
        enable_autocast: bool = True,
        **kwargs,
    ):
        super().__init__()
        if deferred_init:
            return
        self.model = self.create_model()
        self.enable_autocast = enable_autocast
        if device is not None:
            self.model = self.model.to(device)

        self.dtype = dtype
        if dtype is not None:
            self.model = self.model.to(dtype)
            
        if compile:
            self.model = torch.compile(self.model, **compile_kwargs)

    @abstractmethod
    def create_model(self, **kwargs):
        pass

    def pre_transform(self, image: ImArr, **kwargs):
        if isinstance(image, np.ndarray):
            assert image.dtype == np.uint8
            assert image.ndim == 3 or image.ndim == 4
            assert image.shape[-1] == 3
        elif isinstance(image, Tensor):
            assert image.ndim == 4 or image.ndim == 3
            assert image.shape[1] == 3
            assert image.dtype != torch.uint8
        return image

    def post_transform(self, image: ImArr, **kwargs):
        if isinstance(image, np.ndarray):
            assert image.dtype == np.uint8
            image = torch.from_numpy(image) / 255

        assert isinstance(image, Tensor)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.shape[1] > image.shape[-1]: # If have channels last
            image = rearrange(image, "b h w c -> b c h w")

        if image.device == torch.device("cpu"):
            image = image.to(next(iter(self.model.parameters())).device)

        return image

    @abstractproperty
    def transform(self, *args):
        pass

    def reshape_func(self, output_data, **kwargs):
        return output_data

    def forward_model(self, image: Float[Tensor, "b c h w"], **kwargs):
        return self.model(image, **kwargs)

    def validate_input(self, image: ImArr, **kwargs):
        pass

    @torch.no_grad()
    def forward(self, image: ImArr, **kwargs):
        input_data = self.pre_transform(image, **kwargs)
        input_data = self.transform(input_data, **kwargs)
        input_data = self.post_transform(input_data, **kwargs)
        context = torch.autocast(device_type="cuda", dtype=self.dtype) if self.enable_autocast and (input_data.dtype != torch.float32 or self.dtype != torch.float32) else nullcontext()
        with context:
            # Input is a [B, C, H, W] float tensor
            output_data = self.forward_model(input_data, **kwargs)
        return self.reshape_func(output_data, **kwargs)


class TimmModel(BaseModel):
    def __init__(self, model_name: str, num_from_back: int = 0, tensor_input: bool = True, img_size: Optional[tuple[int]] = None, features_only: bool = True, **kwargs):
        self.model_name = model_name
        self.num_from_back = num_from_back
        self.tensor_input = tensor_input
        self.img_size = img_size
        self.features_only = features_only
        super().__init__(**kwargs)

    @functools.cached_property
    def transform(self):
        pretrained_cfg = timm.get_pretrained_cfg(self.model_name, allow_unregistered=False)
        cfg = resolve_data_config(pretrained_cfg=pretrained_cfg.to_dict())
        if hasattr(self, "img_size") and self.img_size is not None:
            cfg["input_size"] = self.img_size
        transform_ = create_transform(**cfg)
        if self.tensor_input:
            transform_.transforms = [x for x in transform_.transforms if not isinstance(x, torchvision.transforms.ToTensor)]
        return transform_
    
    def pre_transform(self, image: ImArr, **kwargs):
        if isinstance(image, (Image.Image, np.ndarray)):
            image = torchvision.transforms.ToTensor()(image)
            
        return image

    def create_model(self, **kwargs):
        if self.img_size is not None:
            kwargs["img_size"] = self.img_size

        if "pretrained" not in kwargs:
            kwargs["pretrained"] = True

        if self.features_only:
            kwargs["features_only"] = True

        return timm.create_model(self.model_name, **kwargs)


class DINOV2(TimmModel):
    def __init__(self, model_name: str = "vit_base_patch14_reg4_dinov2", img_size=(224, 224), **kwargs):
        super().__init__(model_name=model_name, img_size=img_size, features_only=False, **kwargs)

    def pre_transform(self, image: ImArr, **kwargs):
        if isinstance(image, Image.Image):
            image = pad_image_and_adjust_coords(image, patch_size=14)
        
        return super().pre_transform(image, **kwargs)

    def reshape_func(self, output):
        return reshape_vit_output(x=output, num_to_truncate=5)["x"]

    def forward_model(self, image: Float[Tensor, "b c h w"], **kwargs):
        return self.model.forward_features(image, **kwargs)


class VIT(TimmModel):
    def __init__(self, model_name: str = "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k", img_size=(224, 224), **kwargs):
        super().__init__(model_name=model_name, img_size=img_size, features_only=False, **kwargs)

    def pre_transform(self, image: ImArr, **kwargs):
        if isinstance(image, Image.Image):
            image = pad_image_and_adjust_coords(image, patch_size=16)
        return super().pre_transform(image, **kwargs)

    def reshape_func(self, output):
        return reshape_vit_output(x=output, num_to_truncate=1)["x"]

    def forward_model(self, image: Float[Tensor, "b c h w"], **kwargs):
        return self.model.forward_features(image)


class ConvNextV2(TimmModel):
    def __init__(self, model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def reshape_func(self, output: Tensor):
        return get_feature_layer(x=output, num_from_back=self.num_from_back)["x"]

    def forward_model(self, image: Float[Tensor, "b c h w"], **kwargs):
        return self.model.forward(image)


class SwinV2(TimmModel):
    def __init__(self, model_name: str = "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def reshape_func(self, output: Tensor):
        return swin_rearrange(x=output, num_from_back=self.num_from_back)["x"]

    def forward_model(self, image: Float[Tensor, "b c h w"], **kwargs):
        return self.model.forward(image)


class ResNet(TimmModel):
    def __init__(self, model_name: str = "resnet50.fb_ssl_yfcc100m_ft_in1k", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def reshape_func(self, output: torch.Tensor):
        return get_feature_layer(x=output, num_from_back=self.num_from_back)["x"]

    def forward_model(self, image: Float[Tensor, "b c h w"], **kwargs):
        return self.model.forward(image)


class TorchVisionModel(BaseModel):
    def __init__(self, model_builder: Callable, weights, **kwargs):
        self.model_builder = model_builder
        self.weights = weights
        super().__init__(**kwargs)

    def create_model(self):
        return self.model_builder(weights=self.weights)

    def pre_transform(self, image: ImArr, **kwargs):
        if isinstance(image, (Image.Image, np.ndarray)):
            image = torchvision.transforms.ToTensor()(image)
        return image

    def reshape_func(self, output: torch.Tensor):
        return output

    def forward_model(self, image: Float[Tensor, "b c h w"], **kwargs):
        return self.model.forward(image)


class VIT_H_14(TorchVisionModel):
    def __init__(self, img_size=224, **kwargs):
        self.img_size = img_size
        super().__init__(torchvision.models.vit_h_14, torchvision.models.ViT_H_14_Weights.DEFAULT, **kwargs)

    @functools.cached_property
    def transform(self):
        return partial(self.weights.transforms, crop_size=self.img_size, resize_size=self.img_size)()

    def create_model(self):
        model = self.model_builder(weights=self.weights)
        new_model_state = interpolate_embeddings(
            image_size=self.img_size,
            patch_size=14,
            model_state=model.state_dict(),
            interpolation_mode="bicubic",  # Default interpolation mode
            reset_heads=False,  # Whether to reset the heads or not
        )
        model = self.model_builder(image_size=self.img_size)
        model.load_state_dict(new_model_state)
        return model


class ResNet18TorchVision(TorchVisionModel):
    def __init__(self, img_size=224, **kwargs):
        self.img_size = img_size
        super().__init__(torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT, **kwargs)

    @functools.cached_property
    def transform(self):
        return partial(self.weights.transforms, crop_size=self.img_size, resize_size=self.img_size)()


class FeatureExtractorModel(BaseModel):
    def __init__(self, return_nodes, **kwargs):
        self.return_nodes = return_nodes
        super().__init__(**kwargs)

    def create_model(self, **kwargs):
        self.base_model = super().create_model(**kwargs)
        return create_feature_extractor(self.base_model, return_nodes=self.return_nodes)

    def get_nodes(self):
        return get_graph_node_names(self.base_model)
    
    def transform(self, image: ImArr, **kwargs):
        return image

    def forward_model(self, image: ImArr, **kwargs):
        output = self.model(image)

        if self.return_only is not None:
            output = output[self.return_only]
        return output

    def forward_base_model(self, image: ImArr, **kwargs):
        return self.base_model(image)


class ViTFeatureExtractor(FeatureExtractorModel, VIT):
    def __init__(
        self,
        return_nodes={
            "blocks": "blocks",
            "norm": "norm",
            "fc_norm": "fc_norm",
        },
        return_only: Optional[str] = None,
        model_name: str = "vit_base_patch14_reg4_dinov2",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        self.return_only = return_only
        self.pretrained = pretrained
        self.num_classes = num_classes
        super().__init__(model_name=model_name, return_nodes=return_nodes, **kwargs)

    def reshape_func(self, output):
        return output

    def create_model(self):
        create_kwargs = {}
        if self.num_classes is not None:
            create_kwargs["num_classes"] = self.num_classes
        return super().create_model(pretrained=self.pretrained, **create_kwargs)


class ResNetFeatureExtractor(FeatureExtractorModel, ResNet):
    """
    The transforms for this model are currently broken.
    """
    def __init__(
        self,
        return_nodes={
            "layer2": "layer2",
        },
        return_only: Optional[str] = None,
        model_name="resnet18",
        pretrained: bool = True,
        **kwargs,
    ):
        self.return_only = return_only
        self.pretrained = pretrained
        super().__init__(model_name=model_name, return_nodes=return_nodes, **kwargs)

    def reshape_func(self, output):
        return output

    def forward(self, input: torch.Tensor):
        output = super().forward(input)
        return rearrange(output, "b d h w -> b (h w) d")

    def create_model(self):
        return super().create_model(pretrained=self.pretrained)

    @property
    def transform(self):
        transform_ = super().transform
        if self.tensor_input:
            transform_.transforms = [x for x in transform_.transforms if not isinstance(x, torchvision.transforms.Resize)]
        return transform_


def simple_example():
    url = "https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png"
    image = iio.imread(url, index=None)
    image = torch.from_numpy(image).cuda()[:128, :128] / 255
    image = image.permute(2, 0, 1).unsqueeze(0).float()
    model = ResNet18TorchVision().cuda()
    output = model(image)
    print(output.shape)


if __name__ == "__main__":
    simple_example()
