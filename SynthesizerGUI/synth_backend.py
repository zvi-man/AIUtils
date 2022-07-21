from dataclasses import dataclass
from typing import Callable, Tuple, List, Any, Dict
from enum import Enum
from PIL import Image, ImageFilter


class IsActive(Enum):
    ACTIVE = "Active"
    NOT_ACTIVE = "NotActive"

@dataclass
class AugmentationMethod:
    name: str
    func: Callable
    func_argc: Dict[str, Any]
    active: bool = False

    def augment_image(self, pil_im: Image) -> Image:
        return self.func(pil_im, **self.func_argc)


@dataclass
class AugmentationPipe:
    augmentation_list: List[AugmentationMethod]

    def augment_image(self, pil_im: Image) -> Image:
        for aug_method in self.augmentation_list:
            if aug_method.active:
                pil_im = aug_method.augment_image(pil_im)
        return pil_im


class AugmentationUtils:
    @staticmethod
    def blur(input_im: Image, radius: int) -> Image:
        gaussian_filter = ImageFilter.GaussianBlur(radius=radius)
        output_im = input_im.filter(gaussian_filter)
        return output_im
