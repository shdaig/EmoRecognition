import os
import glob
import drowsiness.eeg.eeg as eeg
import drowsiness.qualplot.qualplot as qp
import matplotlib.pyplot as plt
from sixdrepnet import SixDRepNet
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from drowsiness.utils.color_print import printc
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


subjs_dates = {
    "subj_name": ["20240101", "20240102", "20240103"],
}

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

with FaceDetector.create_from_options(options) as detector:
    for subj in subjs_dates:
        for date in subjs_dates[subj]:
            print(f"{subj}_{date}")
            fname_mp4 = glob.glob(f"/home/neuron/mnt/a/A/12proj_sorted/{subj}/{date}/*.mp4")[0]
            print(fname_mp4)

            timestamps_history = []
            pitch_history = []
            yaw_history = []
            roll_history = []

            start_time = time.time()
            cap = cv2.VideoCapture(fname_mp4)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            skip_frames = 9
            while True:
                frame_exists, image = cap.read()
                # print(image.shape)
                if not frame_exists:
                    break
                if skip_frames == 9:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                        pitch = pitch[0]
                        yaw = yaw[0]
                        roll = roll[0]
                    timestamps_history.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                    pitch_history.append(pitch)
                    yaw_history.append(yaw)
                    roll_history.append(roll)
                    skip_frames = 0
                else:
                    skip_frames += 1

            end_time = time.time()
            elapsed_time = end_time - start_time

            df_dict = {
                "timestamp": timestamps_history,
                "pitch": pitch_history,
                "yaw": yaw_history,
                "roll": roll_history
            }

            df_angles_history = pd.DataFrame.from_dict(df_dict)
            df_angles_history.to_csv(f"../angles_dataset_10_frames_fixed/{subj}_{date}.csv", index=False)

            print('Elapsed time: ', elapsed_time)
            printc(f"(!) done {fname_mp4}\n", 'g')
