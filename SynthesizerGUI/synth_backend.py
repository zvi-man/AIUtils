from dataclasses import dataclass, field
from typing import Callable, List, Any, Dict
from enum import Enum

import PIL.Image
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2


class IsActive(Enum):
    ACTIVE = "Active"
    NOT_ACTIVE = "NotActive"


@dataclass
class AugmentationMethod:
    name: str
    func: Callable
    func_argc: Dict[str, Any]
    func_arg_type: Dict[str, type] = field(init=False)
    active: bool = False

    def __post_init__(self):
        self.func_arg_type = dict()
        for arg_name, arg_val in self.func_argc.items():
            self.func_arg_type[arg_name] = type(arg_val)

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

    @staticmethod
    def mirror(input_im: Image) -> Image:
        return ImageOps.mirror(input_im)

    @staticmethod
    def subsample(input_im: Image, resize_factor: float, return_original_size: bool = True) -> Image:
        original_size = input_im.size
        new_size = map(lambda x: round(x * resize_factor), original_size)
        output_im = input_im.resize(new_size)
        if return_original_size:
            output_im = output_im.resize(original_size)
        return output_im

    @staticmethod
    def sharpening(input_im: Image, radius: int) -> Image:
        sharpening_filter = ImageFilter.UnsharpMask(radius=radius)
        output_im = input_im.filter(sharpening_filter)
        return output_im

    @staticmethod
    def motion(input_im: Image, radius: int) -> Image:
        kernel_motion_blur = np.zeros((radius, radius))
        kernel_motion_blur[int((radius - 1) / 2), :] = np.ones(radius)
        kernel_motion_blur = kernel_motion_blur / radius
        input_np_im = np.array(input_im)
        output_np_im = cv2.filter2D(input_np_im, -1, kernel_motion_blur)
        return PIL.Image.fromarray(output_np_im)

    @staticmethod
    def zoom(input_im: Image, top_factor: float, bot_factor: float, left_factor: float, right_factor: float) -> Image:
        original_size = input_im.size
        width, height = original_size
        top_pix = round(top_factor * height)
        bot_pix = height - round(bot_factor * height)
        left_pix = round(left_factor * height)
        right_pix = width - round(right_factor * height)
        cropped_im = input_im.crop((left_pix, top_pix, right_pix, bot_pix))
        return cropped_im.resize(original_size)

    @staticmethod
    def brightness(input_im: Image, brightness_factor: float) -> Image:
        """
        brightness_factor == 1 - image same
        brightness_factor = 0.5 - darkens the image
        brightness_factor = 1.5 - brightens the image
        """
        enhancer = ImageEnhance.Brightness(input_im)
        output_im = enhancer.enhance(brightness_factor)
        return output_im
