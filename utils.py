from torch.hub import download_url_to_file
import torch
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import os

from comfy.model_management import get_torch_device
DEVICE = get_torch_device()

LAMA_MODEL_PATH = Path(__file__).parent.absolute().joinpath("ckpts")
LAMA_CACHE_PATH = Path(__file__).parent.absolute().joinpath("ckpts","cache")
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"

#if img length is not a multiple of 8, then return the length divided by 8
#如果图片长度超过8的倍数,则返回加一后的8的倍数
def fitlength(x) -> int:
    if x % 8 == 0:
        return x
    return int(x // 8 + 1) * 8

# pad image
# 填充图片
def padimage(img):
    # w, h are original image size
    # w, h 是原始图片的大小
    w, h = img.size

    #x, y are padded image size
    # x, y 是填充图片的大小
    x = fitlength(w)
    y = fitlength(h)

    if x!= w or y!= h:
        bgimg = Image.new("RGB", (x, y), (0, 0, 0))
        bgimg.paste(img, (0, 0, w, h))
        return bgimg    
    return img

# pad image
# 填充遮罩
def padmask(img):
    # w, h are original image size
    # w, h 是原始图片的大小
    w, h = img.size

    #x, y are padded image size
    # x, y 是填充图片的大小
    x = fitlength(w)
    y = fitlength(h)

    if x!= w or y!= h:
        bgimg = Image.new("L", (x, y), 0)
        bgimg.paste(img, (0, 0, w, h))
        return bgimg    
    return img

# crop image
# 裁剪图片
def cropimage(img, x, y):
    return img.crop((0, 0, x, y))

# Convert PIL to Tensor
# 图片转张量
def pil2tensor(image, device=DEVICE):
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        raise Exception("Input image should be either PIL Image!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
        print(f"Prepare the imput images")
    elif img.ndim == 2:
        img = img[np.newaxis, ...]
        print(f"Prepare the imput masks")

    assert img.ndim == 3

    try:
        img = img.astype(np.float32) / 255
    except:
        img = img.astype(np.float16) / 255
    
    out_image = torch.from_numpy(img).unsqueeze(0).to(device)
    return out_image

# Tensor to PIL
# 张量转图片
def tensor2pil(image):
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

# pil to comfy
# 图片转comfy格式 (i, 3, w, h) -> (i, h, w, 3)
def pil2comfy(img):
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

# download models
# 下载模型
def get_models_path(filename, url=LAMA_URL, localdir=LAMA_MODEL_PATH):

    model_path = localdir.joinpath(filename)

    if not os.path.exists(model_path):
        print(f"biglama model not found, downloading...")
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            download_url_to_file(url, model_path) 
        except:
            print(f"model download failed, please download it manually")

    return model_path

