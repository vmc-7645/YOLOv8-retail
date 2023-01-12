# Adapted from convert_yolov5.ipynb found here https://www.kaggle.com/datasets/thedatasith/sku110k-annotations?select=convert_yolov5.ipynb

# C:\Users\vinmc\Downloads\SKU110K_fixed>

import cv2
import glob
import os
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to directory where the labeled images were downloaded
dataset_folder  = '/Users/vinmc/Downloads/'

# Github with fixed annotations https://github.com/eg4000/SKU110K_CVPR19
dataset_dirname = 'SKU110K_fixed' # From here https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd
path_images         = Path(dataset_folder) / dataset_dirname / 'images'
path_annotations    = Path(dataset_folder) / dataset_dirname / 'annotations'

prefix_to_channel = {
    "train": "train",
    "val": "validation",
    "test": "test",
}

assert path_images.exists(), f"{path_images} not found"

print("Path exists, getting file names...")

for channel_name in prefix_to_channel.values():
    if not (path_images.parent / channel_name).exists():
        (path_images.parent / channel_name).mkdir()

for path_img in path_images.iterdir():
    for prefix in prefix_to_channel:
        if path_img.name.startswith(prefix):
            path_img.replace(path_images.parent / prefix_to_channel[prefix] / path_img.name)

# Revised list (54 images) courtesy of ankandrew on GitHub:
# https://github.com/eg4000/SKU110K_CVPR19/issues/99#issuecomment-988886374
CORRUPTED_IMAGES = {
    "train": (
        "train_1239.jpg",
        "train_2376.jpg",
        "train_2903.jpg",
        "train_2986.jpg",
        "train_305.jpg",
        "train_3240.jpg",
        "train_340.jpg",
        "train_3556.jpg",
        "train_3560.jpg",
        "train_3832.jpg",
        "train_38.jpg",
        "train_4222.jpg",
        "train_5007.jpg",
        "train_5137.jpg",
        "train_5143.jpg",
        "train_5762.jpg",
        "train_5822.jpg",
        "train_6052.jpg",
        "train_6090.jpg",
        "train_6138.jpg",
        "train_6409.jpg",
        "train_6722.jpg",
        "train_6788.jpg",
        "train_737.jpg",
        "train_7576.jpg",
        "train_7622.jpg",
        "train_775.jpg",
        "train_7883.jpg",
        "train_789.jpg",
        "train_8020.jpg",
        "train_8146.jpg",
        "train_882.jpg",
        "train_903.jpg",
        "train_924.jpg"
    ),
    "validation": (
        "val_147.jpg",
        "val_286.jpg",
        "val_296.jpg",
        "val_386.jpg"
    ),
    "test": (
        "test_132.jpg",
        "test_1346.jpg",
        "test_184.jpg",
        "test_1929.jpg",
        "test_2028.jpg",
        "test_22.jpg",
        "test_2321.jpg",
        "test_232.jpg",
        "test_2613.jpg",
        "test_2643.jpg",
        "test_274.jpg",
        "test_2878.jpg",
        "test_521.jpg",
        "test_853.jpg",
        "test_910.jpg",
        "test_923.jpg"
    ),             
}

print("Fixing corrupted images...")

for channel_name in prefix_to_channel.values():
    for img_name in CORRUPTED_IMAGES[channel_name]:
        try:
            (path_images.parent / channel_name / img_name).unlink()
            print(f"{img_name} removed from channel {channel_name} ")
        except FileNotFoundError:
            print(f"{img_name} not in channel {channel_name}")

################################################################################################
# Run clean after this is finished