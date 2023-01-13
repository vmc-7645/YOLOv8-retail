# YOLOv8-retail
 Detect retail products via the YOLOv8 object recognition engine
# Utilization

Run `pip install -r requirements.txt` to install python dependencies.

Correct version of pytorch (Win10/11) `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

As we are on windows we'll also have to download the correct torchvision

# Training

Heavily inspired by this [article](https://medium.com/analytics-vidhya/retail-store-item-detection-using-yolov5-7ba3ddd71b0c) and this [Kaggle](https://www.kaggle.com/code/thedatasith/visualize-sku110k/notebook), but applied to YOLOv8 instead of YOLOv5 ([GitHub](https://github.com/eg4000/SKU110K_CVPR19) and [model](https://drive.google.com/file/d/1f9tRzJSqjuUQzXz8WjJC0V_WD-8y_6wy/view?usp=sharing) of YOLOv5 trained on same data).

Training data is taken from the SKU110k dataset ([download from kaggle](https://www.kaggle.com/datasets/thedatasith/sku110k-annotations)), which holds several gigabytes of prelabeled images of the subject matter.

After installing CUDA correctly run the following command to begin training:

`yolo task=detect mode=train model=yolov8n.pt data=custom.yaml epochs=300 imgsz=320 workers=4 batch=8`
