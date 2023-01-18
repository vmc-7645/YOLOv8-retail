from ultralytics import YOLO

model = YOLO("models/best18.pt") #Location of PT File
model.export(format="onnx", imgsz=[480,640])