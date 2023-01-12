# Train model by running this file
# Good starting point? https://github.com/eg4000/SKU110K_CVPR19/blob/master/object_detector_retinanet/keras_retinanet/bin/train.py
# Actually go here: https://chr043416.medium.com/train-yolov8-on-custom-data-6d28cd348262

from ultralytics import YOLO

# # Load a pretraind YOLOv8n detection model, train it on COCO128 for 3 epochs and predict an image with it
# from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
model.train(data='custom.yaml', epochs=3)  # train the model

# yolo task=detect mode=train model=yolov8n.pt data=custom.yaml epochs=300 imgsz=640

# task=detect, mode=train, model=yolov8n.yaml, data=coco128.yaml, epochs=100, patience=50, batch=16, imgsz=640, save=True, cache=False, device=None, workers=8, project=None, name=None, exist_ok=False, pretrained=False, optimizer=SGD, verbose=False, seed=0, deterministic=True, single_cls=False, image_weights=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, hide_labels=False, hide_conf=False, vid_stride=1, line_thickness=3, visualize=False, augment=False, agnostic_nms=False, retina_masks=False, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=17, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, fl_gamma=0.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, hydra={'output_subdir': None, 'run': {'dir': '.'}}, v5loader=False, save_dir=C:\Users\vinmc\Documents\Work\RevelDigital\YOLOv8-retail\runs\detect\train7
# Ultralytics YOLOv8.0.0  Python-3.10.4 torch-1.13.1+cpu CPU

