# YOLOv8-retail
 Detect retail products via the YOLOv8 object recognition engine
 
 Demo: [https://www.youtube.com/watch?v=yIRT5nHoH78](https://www.youtube.com/watch?v=yIRT5nHoH78)
 
# Utilization

## Testing

Go to the correct directory `testing` and run one of the following commands:

`python3 video_object_detection.py` for video

`python3 image_object_detection.py` for image

`python3 webcam_object_detection.py` for webcam 

# Training

Correct version of pytorch (Win10/11) `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

As we are on windows we'll also have to download the correct cuda-combatible versions for torch and torchvision.

Heavily inspired by this [article](https://medium.com/analytics-vidhya/retail-store-item-detection-using-yolov5-7ba3ddd71b0c) and this [Kaggle](https://www.kaggle.com/code/thedatasith/visualize-sku110k/notebook), but applied to YOLOv8 instead of YOLOv5 ([GitHub](https://github.com/eg4000/SKU110K_CVPR19) and [model](https://drive.google.com/file/d/1f9tRzJSqjuUQzXz8WjJC0V_WD-8y_6wy/view?usp=sharing) of YOLOv5 trained on same data).

Training data is taken from the SKU110k dataset ([download from kaggle](https://www.kaggle.com/datasets/thedatasith/sku110k-annotations)), which holds several gigabytes of prelabeled images of the subject matter.

After installing CUDA correctly run the following command to begin training:

`yolo task=detect mode=train model=yolov8n.pt data=custom.yaml epochs=300 imgsz=320 workers=4 batch=8`

# Results

## Field Model(s) (v0.2.0-v0.2.1)
Models with exceptional performance used in the field. Versions 0.2.0-0.2.1 used YOLOv8m, versions 0.2.2-Onwards use YOLOv8l.

Example predictions (mislabeled) from a 0.2.1 run: 

![Mislabeled predictions](model/0.2.1/predictions.png?raw=true "Mislabeled predictions, field model")

## Test Model(s) (v0.1.0-0.1.3)

Model(s) used to test the capabilities of the models in some example scenarios. Used YOLOv8s as base model.

Example predictions (mislabeled) from a (0.1.3) run:

![Mislabeled predictions](model/0.1.3/predictions.png?raw=true "Mislabeled predictions")


## Preliminary Model(s) (v0.0.1-0.0.5)

Model(s) used to test whether it was possible to actually train on this dataset. Used YOLOv8n as base model.

Our findings were somewhat dissatisfactory when it came to the actual results of the training, however they did result in some models that were not completely useless. Thus we went on to invest more resources into training better models.

Example predictions form a (0.0.1) run:

![Valid Batch 2](model/0.0.1/val_batch2_pred.jpg?raw=true "Valid Batch 2 Predictions")

## Inverse

Model that instead of getting the retail objects, gets the empty shelves where retail items should be. 

[Download Link](https://drive.google.com/drive/folders/1ijsV8MYNFnP8mauivFEEQ4h2IwJ3HHT7?usp=sharing)