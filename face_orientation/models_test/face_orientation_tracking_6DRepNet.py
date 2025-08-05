import cv2
import numpy as np
from sixdrepnet import SixDRepNet

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import warnings

warnings.filterwarnings("ignore")


def get_bbox_points(bbox, img_shape):
    img_h, img_w, img_c = img_shape
    k = 1.5
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    center_point = [int(x + w / 2), int(y + h / 2)]
    start_point = [int(center_point[0] - (w / 2 * k)),
                   int(center_point[1] - (h / 2 * (k + 0.5)))]
    end_point = [int(center_point[0] + (w / 2 * k)), int(center_point[1] + (h / 2 * k))]
    start_point[0] = np.clip(start_point[0], 0, img_w - 1)
    start_point[1] = np.clip(start_point[1], 0, img_h - 1)
    end_point[0] = np.clip(end_point[0], 0, img_w - 1)
    end_point[1] = np.clip(end_point[1], 0, img_h - 1)
    return center_point, start_point, end_point


cap = cv2.VideoCapture(0)
model = SixDRepNet()

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../../models/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

tracker = cv2.TrackerKCF_create()
tracker_inited = False
prev_bbox = []

with FaceDetector.create_from_options(options) as detector:
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)
        detected = False
        pitch, yaw, roll = np.nan, np.nan, np.nan
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)
        if len(detection_result.detections) > 0:
            detection = detection_result.detections[0]
            bbox = detection.bounding_box
            cur_bbox = [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height]
            tracker = cv2.TrackerKCF_create()
            tracker.init(image, cur_bbox)
            tracker_inited = True
            center_point, start_point, end_point = get_bbox_points(cur_bbox, image.shape)
            detected = True
        elif tracker_inited:
            success, bbox = tracker.update(image)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                center_point, start_point, end_point = get_bbox_points((x, y, w, h), image.shape)
                detected = True
            else:
                tracker_inited = False

        if detected:
            pitch, yaw, roll = model.predict(image[start_point[1]:end_point[1], start_point[0]:end_point[0]])
            model.draw_axis(image, yaw, pitch, roll, center_point[0], center_point[1])
            cv2.rectangle(image, start_point, end_point, (255, 0, 0), 3)
            print(pitch[0], yaw[0], roll[0])
        else:
            print("no detection")

        cv2.imshow('live cam', image)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
