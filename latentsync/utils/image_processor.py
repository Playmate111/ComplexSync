# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from util.util import create_masks_from_landmarks_box


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

    def outward_blur_mask(self, mask):
        dilated_mask = ((1 - mask)[0].numpy() * 255).astype(np.uint8)
        blur_radius = 10
        kernel_size = blur_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(dilated_mask, kernel)
        blurred_mask = cv2.GaussianBlur(dilated_mask, (kernel_size, kernel_size), blur_radius)
        blurred_mask = 1 - torch.from_numpy(blurred_mask / 255).unsqueeze(0)

        return blurred_mask

    def preprocess_fixed_mask_image(self, image: torch.Tensor, lm=None, need_blur=False):
        org_h, org_w = image.shape[-2:]
        image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        if lm is not None:
            lm[..., 0] = lm[..., 0] / org_w * self.resolution
            lm[..., 1] = lm[..., 1] / org_h * self.resolution
            mask = 1 - create_masks_from_landmarks_box(lm[None, :, :], (self.resolution, self.resolution))
            masked_pixel_values = pixel_values * mask
            if need_blur:
                blur_mask = self.outward_blur_mask(mask)
                return pixel_values, masked_pixel_values, mask, blur_mask
            else:
                return pixel_values, masked_pixel_values, mask
        else:
            masked_pixel_values = pixel_values * self.mask_image
            if need_blur:
                blur_mask = self.outward_blur_mask(self.mask_image)
                return pixel_values, masked_pixel_values, self.mask_image[0:1], blur_mask
            else:
                return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(
        self, images: Union[torch.Tensor, np.ndarray], lms=None, need_blur=False
    ):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        if lms is not None:
            results = [
                self.preprocess_fixed_mask_image(image, lm=lm, need_blur=need_blur)
                for image, lm in zip(images, lms)
            ]
        else:
            results = [
                self.preprocess_fixed_mask_image(image, lm=None, need_blur=need_blur)
                for image in images
            ]

        if need_blur:
            pixel_values_list, masked_pixel_values_list, masks_list, blur_mask_list = list(zip(*results))
            return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), \
                torch.stack(masks_list), torch.stack(blur_mask_list)
        else:
            pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
            return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)
