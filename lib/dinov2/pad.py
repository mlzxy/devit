import torch
import math
import detectron2.data.transforms as T
from fvcore.transforms.transform import (
    PadTransform,
    Transform,
    TransformList
)

from detectron2.data.transforms.augmentation import Augmentation

class SizeDivisibilityPad(Augmentation):

    @torch.jit.unused
    def __init__(self, divide_by=14):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()
        self.divide_by = divide_by

    @torch.jit.unused
    def get_transform(self, image):
        h, w = image.shape[:2]
        py = int(math.ceil(h / self.divide_by)) * self.divide_by - h
        px = int(math.ceil(w / self.divide_by)) * self.divide_by - w

        py0 = py // 2
        py1 = py - py0

        px0 = px // 2
        px1 = px - px0
        
        return PadTransform(px0, py0, px1, py1)