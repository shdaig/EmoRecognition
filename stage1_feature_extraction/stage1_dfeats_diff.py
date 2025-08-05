import glob

from drowsiness.emo import dfeats_utils as dfu

import cv2 as cv
import pandas as pd
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../models/face_landmarker.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)

face_blendshapes_names = dfu.get_features_names()
blendshapes_dict = dfu.get_features_dict()

neutral_frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/neutral/*_1.png"))
disgust_frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/disgust/*_1.png"))
smile_frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/smile/*_1.png"))

with FaceLandmarker.create_from_options(options) as landmarker:
    for frame_1_filename in neutral_frames_path_1:
        frame_2_filename = frame_1_filename[:-5] + '2.png'
        face_frame_1 = cv.imread(frame_1_filename)
        face_frame_2 = cv.imread(frame_2_filename)
        mp_image_1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame_1)
        mp_image_2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame_2)
        face_landmarker_result_1 = landmarker.detect(mp_image_1)
        face_landmarker_result_2 = landmarker.detect(mp_image_2)
        if len(face_landmarker_result_1.face_landmarks) != 0 and len(face_landmarker_result_2.face_landmarks) != 0:
            face_blendshapes_scores_1 = dfu.get_face_landmarks_features(face_landmarker_result_1.face_landmarks[0])
            face_blendshapes_scores_2 = dfu.get_face_landmarks_features(face_landmarker_result_2.face_landmarks[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(
                    face_blendshapes_scores_2[i] - face_blendshapes_scores_1[i])
            blendshapes_dict["filename"].append(frame_1_filename)
            blendshapes_dict["label"].append(0)

    for frame_1_filename in smile_frames_path_1:
        frame_2_filename = frame_1_filename[:-5] + '2.png'
        face_frame_1 = cv.imread(frame_1_filename)
        face_frame_2 = cv.imread(frame_2_filename)
        mp_image_1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame_1)
        mp_image_2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame_2)
        face_landmarker_result_1 = landmarker.detect(mp_image_1)
        face_landmarker_result_2 = landmarker.detect(mp_image_2)
        if len(face_landmarker_result_1.face_landmarks) != 0 and len(face_landmarker_result_2.face_landmarks) != 0:
            face_blendshapes_scores_1 = dfu.get_face_landmarks_features(face_landmarker_result_1.face_landmarks[0])
            face_blendshapes_scores_2 = dfu.get_face_landmarks_features(face_landmarker_result_2.face_landmarks[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(
                    face_blendshapes_scores_2[i] - face_blendshapes_scores_1[i])
            blendshapes_dict["filename"].append(frame_1_filename)
            blendshapes_dict["label"].append(1)

    for frame_1_filename in disgust_frames_path_1:
        frame_2_filename = frame_1_filename[:-5] + '2.png'
        face_frame_1 = cv.imread(frame_1_filename)
        face_frame_2 = cv.imread(frame_2_filename)
        mp_image_1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame_1)
        mp_image_2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame_2)
        face_landmarker_result_1 = landmarker.detect(mp_image_1)
        face_landmarker_result_2 = landmarker.detect(mp_image_2)
        if len(face_landmarker_result_1.face_landmarks) != 0 and len(face_landmarker_result_2.face_landmarks) != 0:
            face_blendshapes_scores_1 = dfu.get_face_landmarks_features(face_landmarker_result_1.face_landmarks[0])
            face_blendshapes_scores_2 = dfu.get_face_landmarks_features(face_landmarker_result_2.face_landmarks[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(
                    face_blendshapes_scores_2[i] - face_blendshapes_scores_1[i])
            blendshapes_dict["filename"].append(frame_1_filename)
            blendshapes_dict["label"].append(2)

blendshapes_df = pd.DataFrame.from_dict(blendshapes_dict)
print(blendshapes_df.head(5))
blendshapes_df.to_csv("dfeats_diff.csv", index=False)



