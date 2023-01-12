
################################################################################################

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

# Expected output:
# Number of train images = 8185
# Number of validation images = 584
# Number of test images = 2919
for channel_name in prefix_to_channel.values():
    print(f"Number of {channel_name} images = {sum(1 for x in (path_images.parent / channel_name).glob('*.jpg'))}")

# os.rmdir(path_images)

yolov5_dataset_folder = os.getcwd()
yolov5_sku_dataset_dirname = 'SKU110K_fixed'
# local_path_annotations = Path(yolov5_dataset_folder) / yolov5_sku_dataset_dirname / 'labels'
# local_path_images = Path(yolov5_dataset_folder) / yolov5_sku_dataset_dirname / 'images'
local_path_annotations = path_annotations
local_path_images = path_images



names = 'image', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'
annotation_files =  path_annotations.glob('*.csv')

print("Getting annotation...")

import sys

# \Users\vinmc\Downloads\SKU110K_fixed\annotations\annotations_test.csv
# Traceback (most recent call last):
#   File "C:\Users\vinmc\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196, in _run_module_as_main
#     return _run_code(code, main_globals, None,
#   File "C:\Users\vinmc\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86, in _run_code
#     exec(code, run_globals)
#   File "C:\Users\vinmc\Documents\Work\RevelDigital\YOLOv8-retail\training\yolov8-convert.py", line 146, in <module>
#     os.mkdir(out_labels_dir)
# FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Users\\vinmc\\Documents\\Work\\RevelDigital\\YOLOv8-retail\\training\\SKU110K_fixed\\labels\\test'

os.chdir(dataset_folder)

for file in annotation_files:

    print("File "+str(file))
    data = pd.read_csv(file, names=names)  # annotations
    prefix = file.name.split('_')[-1].replace('.csv','')
    out_labels_dir = local_path_annotations / prefix
    out_images_dir = local_path_images / prefix

    isExist = os.path.exists(out_labels_dir)
    if not isExist:
        os.mkdir(out_labels_dir)

    isExist = os.path.exists(out_images_dir)
    if not isExist:
        os.mkdir(out_images_dir)
        
    for filename_img in data['image'].unique():
        # Get all bounding boxes for this image
        mask_filename_img = data['image'] == filename_img
        data_img = data[mask_filename_img].copy().reset_index()

        # Reformat each bounding box and add it to output file
        # YOLO format is normalized (img_width, img_height) = (1, 1)
        # NOTE: there are several erroneous annotations. Please see: 
        # https://github.com/eg4000/SKU110K_CVPR19/issues
        # for details.
        # 
        # I noticed a quite a few bounding boxes exceeding the boundaries
        # of the image. Ideally I should do something more sophisticated, 
        # but per eye inspection, the differences were "negligible". In 
        # order to be able to use the infringing image/annotation pairs 
        # when training a YOLOv5 model (which checks the normalization 
        # range to be [0,1]), I'm simply clipping to an upper bound of 1.
        im_width = data_img.image_width[0]
        im_height = data_img.image_height[0]

        data_img['width'] = data_img['x2'] - data_img['x1']
        data_img['height'] = data_img['y2'] - data_img['y1']
        data_img['xc'] = data_img['x1'] + data_img['width']/2
        data_img['yc'] = data_img['y1'] + data_img['height']/2
        
        data_img['xc'] = data_img['xc'] / im_width
        data_img['yc'] = data_img['yc'] / im_height
        data_img['width'] = data_img['width'] / im_width
        data_img['height'] = data_img['height'] / im_height


        data_img['xc'] = data_img['xc'].where(data_img['xc'] <= 1., 1.) 
        data_img['yc'] = data_img['yc'].where(data_img['yc'] <= 1., 1.) 
        data_img['width'] = data_img['width'].where(data_img['width'] <= 1., 1.) 
        data_img['height'] = data_img['height'].where(data_img['height'] <= 1., 1.) 
        data_img['class'] = 0
        
        data_img = data_img[['class','xc','yc','width','height']]

        # Set up the necessary paths
        filename_label = filename_img.replace('jpg','txt')
        out_labels_file = out_labels_dir / filename_label
        in_images_file = path_images.parent / prefix_to_channel[prefix] / filename_img
        out_images_file = out_images_dir / filename_img

        try:
            _ = shutil.copy2(in_images_file, out_images_file)
        except:
            # Exceptions are due to image file not existing for the corresponding label
            # raise NameError('check the image file name')
            print(f'check the image file name {filename_img}') 
            continue

        # If the image file is found and copied, it's safe to generate the corresponding label file
        data_img.to_csv(out_labels_file , sep=' ', header=False, index=False)

from PIL import ExifTags, Image, ImageOps

# Include image suffixes
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break  

def exif_size(img):  
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass
    return s

for prefix in prefix_to_channel:
    _path = local_path_images / prefix / '*'
    local_files_images = glob.glob(_path.as_posix())
    for idx in range(0, len(local_files_images)):
        im_file = local_files_images[idx]
        im = Image.open(im_file) 
        im.verify()  # PIL verify 
        shape = exif_size(im)  # image size 
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels' 
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}' 
        if im.format.lower() in ('jpg', 'jpeg'): 
            with open(im_file, 'rb') as f: 
                f.seek(-2, 2) 
                if f.read() != b'\xff\xd9':  # corrupt JPEG 
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100) 
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved' 
                    print(msg)


# Visualize some results
counter = 1
plt.figure(figsize=(20, 20), facecolor='white')

for prefix in prefix_to_channel:
    _path = local_path_images / prefix / '*'
    local_files_images = glob.glob(_path.as_posix())
    for idx in random.sample(range(0, len(local_files_images)), 3):
        filename_image = local_files_images[idx]
        filename_label = filename_image.replace('images','labels').replace('jpg','txt')
        data = pd.read_csv(filename_label, header=None, delimiter=' ')
        
        print(filename_image)
        print(filename_label)
        im = cv2.imread(filename_image)
        im_size = im.shape[:2]
        for _, bbox in data.iterrows():
            cls, xc, yc, w, h = bbox
            xmin = xc - w/2
            ymin = yc - h/2
            xmax = xc + w/2
            ymax = yc + h/2

            xmin *= im_size[1]
            ymin *= im_size[0]
            xmax *= im_size[1]
            ymax *= im_size[0]

            start_point = (int(xmin), int(ymin))
            end_point = (int(xmax), int(ymax))
            color = (0, 100, 175)
            thickness = 10

            im = cv2.rectangle(im, start_point, end_point, color, thickness)

        ax = plt.subplot(3, 3, counter)
        plt.title(prefix)
        plt.axis("off")
        plt.imshow(im)
        counter += 1

##############################################
# Once done were training the data...