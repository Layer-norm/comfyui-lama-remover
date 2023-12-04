# reference from https://github.com/enesmsahin/simple-lama-inpainting

import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from ..utils import get_models_path
from comfy.model_management import get_torch_device
DEVICE = get_torch_device()

class BigLama:

    def __init__(self):
        self.device = DEVICE
        model_path = get_models_path(filename="big-lama.pt")

        print(f"{model_path}")

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
        except:
            print(f"can't use comfy device")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.jit.load(model_path, map_location=self.device)
        
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image, mask):

        with torch.inference_mode():
            result = self.model(image, mask)

            return result[0]
