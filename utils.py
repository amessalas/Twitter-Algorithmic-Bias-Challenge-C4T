import os

import pandas as pd
from external import ImageSaliencyModel
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import trange, tqdm
import cv2
import platform
from typing import List

IMGS_ROOT = 'data/fairface-img-margin125-trainval'

BIN_MAPS = {"Darwin": "mac", "Linux": "linux"}
HOME_DIR = Path("./").expanduser()
bin_dir = HOME_DIR / Path("./bin")
bin_path = bin_dir / BIN_MAPS[platform.system()] / "candidate_crops"
model_path = bin_dir / "fastgaze.vxm"
np.random.seed(0)


def get_data():
    df_train = pd.read_csv("data/fairface_label_train.csv")
    df_val = pd.read_csv("data/fairface_label_val.csv")
    df = pd.concat([df_train, df_val])
    df.file = df.file.transform(lambda x: f"{IMGS_ROOT}/{x}")
    return df


def create_collage_image(img1: Image, img2: Image):
    width = 448
    height = 1123
    new_image = Image.new('RGB', (width, height), (255, 255, 255))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, 675))
    return new_image


def comparison(group1: List[str], group2: List[str], n_samples: int):
    chosen_group1, chosen_group2 = [], []
    os.makedirs('data/collages', exist_ok=True)
    for i, _ in enumerate(trange(n_samples)):
        img1_path = np.random.choice(group1)
        img2_path = np.random.choice(group2)

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img = create_collage_image(img1, img2)
        img.save(f'data/collages/{i}.png')
        img_path = Path(f'data/collages/{i}.png')

        model = ImageSaliencyModel(crop_binary_path=bin_path, crop_model_path=model_path)
        salient_x, salient_y = model.get_saliency_point(img_path=img_path)
        print(salient_y)
        if salient_y < 448:
            chosen_group1.append(img1_path)
        elif salient_y > 675:
            chosen_group2.append(img2_path)
        else:
            continue
    return chosen_group1, chosen_group2


def contrast(image: Image):
    if Image.isImageType(image):
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_img.std()


def sharpness(image: Image):
    if Image.isImageType(image):
        image = np.array(image).astype(np.uint8)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def measure_image_properties(images: List['str']):
    c, s = [], []
    with tqdm(total=len(images)) as pbar:
        for img_name in images:
            img = Image.open(img_name)
            c.append(contrast(img))
            s.append(sharpness(img))
            pbar.update(1)
        df_properties = pd.DataFrame([c, s], index=['contrast', 'index'], columns=images).T
        return df_properties