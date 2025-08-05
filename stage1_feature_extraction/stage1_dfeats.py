import glob

from drowsiness.emo import dfeats_utils as dfu

import cv2 as cv
import pandas as pd
import mediapipe as mp


def get_face_landmarks_features(face_landmarks):
    data = face_landmarks

    nose_dist = dfu.nose_dist_calc(data)
    center_right_mouth_calc = dfu.center_right_mouth_calc(data, nose_dist)
    center_left_mouth_calc = dfu.center_left_mouth_calc(data, nose_dist)
    nose_right_mouth_calc = dfu.nose_right_mouth_calc(data, nose_dist)
    nose_left_mouth_calc = dfu.nose_left_mouth_calc(data, nose_dist)
    horizontal_mouth_calc = dfu.horizontal_mouth_calc(data, nose_dist)
    vertical_mouth_calc = dfu.vertical_mouth_calc(data, nose_dist)
    nose_upper_mouth_calc = dfu.nose_upper_mouth_calc(data, nose_dist)
    nose_lower_mouth_calc = dfu.nose_lower_mouth_calc(data, nose_dist)
    nose_left_upper_lip_calc = dfu.nose_left_upper_lip_calc(data, nose_dist)
    nose_right_upper_lip_calc = dfu.nose_right_upper_lip_calc(data, nose_dist)
    nose_wings_calc = dfu.nose_wings_calc(data, nose_dist)
    inner_brows_calc = dfu.inner_brows_calc(data, nose_dist)
    nose_left_brow_calc = dfu.nose_left_brow_calc(data, nose_dist)
    nose_right_brow_calc = dfu.nose_right_brow_calc(data, nose_dist)
    left_cheek_eye = dfu.left_cheek_eye(data, nose_dist)
    right_cheek_eye = dfu.right_cheek_eye(data, nose_dist)

    face_blendshapes_scores = [center_right_mouth_calc, center_left_mouth_calc,
                               nose_right_mouth_calc, nose_left_mouth_calc,
                               horizontal_mouth_calc, vertical_mouth_calc,
                               nose_upper_mouth_calc, nose_lower_mouth_calc,
                               nose_left_upper_lip_calc, nose_right_upper_lip_calc,
                               nose_wings_calc, inner_brows_calc,
                               nose_left_brow_calc, nose_right_brow_calc,
                               left_cheek_eye, right_cheek_eye]
    return face_blendshapes_scores



BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../models/face_landmarker.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)

face_blendshapes_names = ['center_right_mouth_calc', 'center_left_mouth_calc',
                          'nose_right_mouth_calc', 'nose_left_mouth_calc',
                          'horizontal_mouth_calc', 'vertical_mouth_calc',
                          'nose_upper_mouth_calc', 'nose_lower_mouth_calc',
                          'nose_left_upper_lip_calc', 'nose_right_upper_lip_calc',
                          'nose_wings_calc', 'inner_brows_calc',
                          'nose_left_brow_calc', 'nose_right_brow_calc',
                          'left_cheek_eye', 'right_cheek_eye']

blendshapes_dict = {
    "filename": [], "label": [],
    'center_right_mouth_calc': [], 'center_left_mouth_calc': [],
    'nose_right_mouth_calc': [], 'nose_left_mouth_calc': [],
    'horizontal_mouth_calc': [], 'vertical_mouth_calc': [],
    'nose_upper_mouth_calc': [], 'nose_lower_mouth_calc': [],
    'nose_left_upper_lip_calc': [], 'nose_right_upper_lip_calc': [],
    'nose_wings_calc': [], 'inner_brows_calc': [],
    'nose_left_brow_calc': [], 'nose_right_brow_calc': [],
    'left_cheek_eye': [], 'right_cheek_eye': []
}

neutral_frames_path = sorted(glob.glob("face_emo_datasets/frames_filtered/neutral/*_2.png"))
disgust_frames_path = sorted(glob.glob("face_emo_datasets/frames_filtered/disgust/*_2.png"))
smile_frames_path = sorted(glob.glob("face_emo_datasets/frames_filtered/smile/*_2.png"))

with FaceLandmarker.create_from_options(options) as landmarker:
    for frame_filename in neutral_frames_path:
        face_frame = cv.imread(frame_filename)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
        face_landmarker_result = landmarker.detect(mp_image)
        if len(face_landmarker_result.face_landmarks) != 0:
            face_landmarks_scores = get_face_landmarks_features(face_landmarker_result.face_landmarks[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(face_landmarks_scores[i])
            blendshapes_dict["filename"].append(frame_filename)
            blendshapes_dict["label"].append(0)

    for frame_filename in smile_frames_path:
        face_frame = cv.imread(frame_filename)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
        face_landmarker_result = landmarker.detect(mp_image)
        if len(face_landmarker_result.face_landmarks) != 0:
            face_landmarks_scores = get_face_landmarks_features(face_landmarker_result.face_landmarks[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(face_landmarks_scores[i])
            blendshapes_dict["filename"].append(frame_filename)
            blendshapes_dict["label"].append(1)

    for frame_filename in disgust_frames_path:
        face_frame = cv.imread(frame_filename)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
        face_landmarker_result = landmarker.detect(mp_image)
        if len(face_landmarker_result.face_landmarks) != 0:
            face_landmarks_scores = get_face_landmarks_features(face_landmarker_result.face_landmarks[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(face_landmarks_scores[i])
            blendshapes_dict["filename"].append(frame_filename)
            blendshapes_dict["label"].append(2)

blendshapes_df = pd.DataFrame.from_dict(blendshapes_dict)
print(blendshapes_df.head(5))
blendshapes_df.to_csv("dfeats.csv", index=False)



