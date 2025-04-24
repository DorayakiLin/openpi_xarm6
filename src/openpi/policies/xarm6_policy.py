import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_xarm6_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(1080, 1920, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """
    将图像转换为 uint8 格式，并确保为 HWC 格式。
    支持从 float32 (C,H,W) → uint8 (H,W,C) 自动转换。
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class xarm6Inputs(transforms.DataTransformFn):
    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        if "actions" in data:

            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        
        inputs["prompt"] = "pick up the bottle and put it in the box"

        return inputs
    

@dataclasses.dataclass(frozen=True)
class xarm6Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
