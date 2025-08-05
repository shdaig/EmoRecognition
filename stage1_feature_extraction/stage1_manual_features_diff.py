import glob

from drowsiness.emo import feat_utils as fu

import cv2 as cv
import pandas as pd
import mediapipe as mp


def get_face_landmarks_features(face_landmarks):
    data = [face_landmarks]

    nose_dist = fu.nose_dist_calc(data)
    corner_dist_nose = fu.corner_dist_nose_calc(data, nose_dist)[0]
    corner_dist = fu.corner_dist_calc(data, nose_dist)[0]
    mouth_up_down_dist = fu.mouth_up_down_dist_calc(data, nose_dist)[0]
    left_right_nose = fu.left_right_nose_calc(data, nose_dist)[0]
    up_mouth = fu.up_mouth_calc(data, nose_dist)[0]
    noses_fly = fu.noses_fly_calc(data, nose_dist)[0]
    eye = fu.eye_calc(data, nose_dist)[0]
    brov = fu.brov_calc(data, nose_dist)[0]
    new_feat_mouth_updown = fu.new_feat_mouth_updown(data, nose_dist)[0]

    face_blendshapes_scores = [corner_dist_nose, corner_dist, mouth_up_down_dist, left_right_nose, up_mouth, noses_fly, eye, brov, new_feat_mouth_updown]
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

face_blendshapes_names = ['corner_dist_nose', 'corner_dist', 'mouth_up_down_dist', 'left_right_nose', 'up_mouth',
                          'noses_fly', 'eye', 'brov', 'new_feat_mouth_updown']

blendshapes_dict = {
    "filename": [], "label": [],
    'corner_dist_nose': [], 'corner_dist': [], 'mouth_up_down_dist': [], 'left_right_nose': [], 'up_mouth': [],
    'noses_fly': [], 'eye': [], 'brov': [], 'new_feat_mouth_updown': []
}

neutral_frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/neutral/*_1.png"))
disgust_frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/disgust/*_1.png"))
smile_frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/smile/*_1.png"))

neutral_frames_path_2 = sorted(glob.glob("face_emo_datasets/frames_filtered/neutral/*_2.png"))
disgust_frames_path_2 = sorted(glob.glob("face_emo_datasets/frames_filtered/disgust/*_2.png"))
smile_frames_path_2 = sorted(glob.glob("face_emo_datasets/frames_filtered/smile/*_2.png"))

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
            face_blendshapes_scores_1 = get_face_landmarks_features(face_landmarker_result_1.face_landmarks[0])
            face_blendshapes_scores_2 = get_face_landmarks_features(face_landmarker_result_2.face_landmarks[0])
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
            face_blendshapes_scores_1 = get_face_landmarks_features(face_landmarker_result_1.face_landmarks[0])
            face_blendshapes_scores_2 = get_face_landmarks_features(face_landmarker_result_2.face_landmarks[0])
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
            face_blendshapes_scores_1 = get_face_landmarks_features(face_landmarker_result_1.face_landmarks[0])
            face_blendshapes_scores_2 = get_face_landmarks_features(face_landmarker_result_2.face_landmarks[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(
                    face_blendshapes_scores_2[i] - face_blendshapes_scores_1[i])
            blendshapes_dict["filename"].append(frame_1_filename)
            blendshapes_dict["label"].append(2)

blendshapes_df = pd.DataFrame.from_dict(blendshapes_dict)
print(blendshapes_df.head(5))
blendshapes_df.to_csv("frames_diff_manual_filtered_feats.csv", index=False)



