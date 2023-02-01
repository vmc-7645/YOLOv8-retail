import cv2
# from imread_from_url import imread_from_url
from yolov8 import YOLOv8

#Initialize yolov8
model_path = "models/best19.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.45, iou_thres=0.5)

#Read image
# img_url = "https://media.newyorker.com/photos/5e5ed01f39e0e500082b73b6/master/w_2560%2Cc_limit/Rosner-CoronavirusPanicShopping.jpg"
# img = imread_from_url(img_url)
img = cv2.imread("testimg.jpg")

#Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

#Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("test/images/detected_objects.jpg", combined_img)
cv2.waitKey(0)
