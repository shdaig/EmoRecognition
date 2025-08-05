import cv2
import numpy as np
from sixdrepnet import SixDRepNet

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import warnings
warnings.filterwarnings("ignore")

cap = cv2.VideoCapture(0)
model = SixDRepNet()

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../../blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

with FaceDetector.create_from_options(options) as detector:
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)
        img_h, img_w, img_c = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            k = 1.5
            center_point = [int(bbox.origin_x + bbox.width / 2), int(bbox.origin_y + bbox.height / 2)]
            start_point = [int(center_point[0] - (bbox.width / 2 * k)), int(center_point[1] - (bbox.height / 2 * (k + 0.5)))]
            end_point = [int(center_point[0] + (bbox.width / 2 * k)), int(center_point[1] + (bbox.height / 2 * k))]

            start_point[0] = np.clip(start_point[0], 0, img_w-1)
            start_point[1] = np.clip(start_point[1], 0, img_h-1)
            end_point[0] = np.clip(end_point[0], 0, img_w-1)
            end_point[1] = np.clip(end_point[1], 0, img_h-1)
            # print(f"{start_point[0]}:{end_point[0]}, {start_point[1]}:{end_point[1]}")
            pitch, yaw, roll = model.predict(image[start_point[1]:end_point[1], start_point[0]:end_point[0]])

            cv2.rectangle(image, start_point, end_point, (255, 0, 0), 3)
            model.draw_axis(image, yaw, pitch, roll, center_point[0], center_point[1])
            # image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

        cv2.imshow('live cam', image)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
