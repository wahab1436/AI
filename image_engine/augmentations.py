import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

class TrainingAugmentations:
    def __init__(
        self,
        brightness_range: tuple = (0.9, 1.1),
        contrast_range: tuple = (0.9, 1.1),
        crop_range: tuple = (0.9, 1.0),
        noise_sigma: float = 2.0,
        flip_prob: float = 0.5
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.crop_range = crop_range
        self.noise_sigma = noise_sigma
        self.flip_prob = flip_prob
        
    def apply(self, image: Image.Image) -> Image.Image:
        img = image.copy()
        
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        brightness = random.uniform(*self.brightness_range)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        
        contrast = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        
        if random.random() < 0.5:
            crop_factor = random.uniform(*self.crop_range)
            w, h = img.size
            new_w, new_h = int(w * crop_factor), int(h * crop_factor)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))
            img = img.resize((w, h), Image.BILINEAR)
            
        if self.noise_sigma > 0:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.noise_sigma, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            
        return img
