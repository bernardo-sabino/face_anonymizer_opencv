import cv2
import os 
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision 
import time

CAMERA_INDEX = 0
MODEL = "blaze_face_short_range.tflite"

webcam = cv2.VideoCapture(CAMERA_INDEX)
if not webcam.isOpened():
    print("There was a problem trying to open the camera. Closing the program...")
    exit()

model_path = os.path.join('.', 'models', MODEL)
base_options = mpt.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options,
                                     running_mode=vision.RunningMode.VIDEO)
detector = vision.FaceDetector.create_from_options(options)

while True:
    ret, frame = webcam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms=int(time.time() * 1000))
    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            h_img, w_img, _ = frame.shape
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w_img, bbox.origin_x + bbox.width)
            y2 = min(h_img, bbox.origin_y + bbox.height)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                k_size = int(roi.shape[1] * 0.45)
                if k_size % 2 == 0: k_size+=1
                k_size = max(1, k_size)
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k_size,k_size), 0)

    cv2.imshow("Face Blurring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.close()
webcam.release()
cv2.destroyAllWindows()






