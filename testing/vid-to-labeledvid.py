import cv2
# from cap_from_youtube import cap_from_youtube
from yolov8 import YOLOv8

#MUST USE: python3 -m pip install protobuf==3.20.3

#Initialize video
cap = cv2.VideoCapture("test/test.mp4")
#cap = cv2.VideoCapture('test/images/myFridge.mp4')
start_time = 40 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('testout.mp4', fourcc, float(cap.get(cv2.CAP_PROP_FPS)), (frame_width,frame_height))


#Initialize YOLOv8 model
model_path = "models/best19.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.35, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

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

    out.write(combined_img)
    
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()