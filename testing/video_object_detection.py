import cv2
from cap_from_youtube import cap_from_youtube
from yolov8 import YOLOv8

#MUST USE: python3 -m pip install protobuf==3.20.3

#Initialize video
videoUrl = 'https://youtu.be/5LnqliAfaQ4'
cap = cap_from_youtube(videoUrl, resolution='720p')
#cap = cv2.VideoCapture('test/images/myFridge.mp4')
start_time = 5 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

#Initialize YOLOv8 model
model_path = "models/best18.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    #Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        #Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)