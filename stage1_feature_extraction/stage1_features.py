import glob

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2 as cv
import pandas as pd


def get_face_blendshapes_scores(face_blendshapes):
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
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

face_blendshapes_names = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
                          'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft',
                          'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
                          'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
                          'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen',
                          'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
                          'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
                          'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
                          'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
                          'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
                          'noseSneerLeft', 'noseSneerRight']

blendshapes_dict = {
    "filename": [], "label": [],
    '_neutral': [], 'browDownLeft': [], 'browDownRight': [], 'browInnerUp': [], 'browOuterUpLeft': [],
    'browOuterUpRight': [], 'cheekPuff': [], 'cheekSquintLeft': [], 'cheekSquintRight': [], 'eyeBlinkLeft': [],
    'eyeBlinkRight': [], 'eyeLookDownLeft': [], 'eyeLookDownRight': [], 'eyeLookInLeft': [], 'eyeLookInRight': [],
    'eyeLookOutLeft': [], 'eyeLookOutRight': [], 'eyeLookUpLeft': [], 'eyeLookUpRight': [], 'eyeSquintLeft': [],
    'eyeSquintRight': [], 'eyeWideLeft': [], 'eyeWideRight': [], 'jawForward': [], 'jawLeft': [], 'jawOpen': [],
    'jawRight': [], 'mouthClose': [], 'mouthDimpleLeft': [], 'mouthDimpleRight': [], 'mouthFrownLeft': [],
    'mouthFrownRight': [], 'mouthFunnel': [], 'mouthLeft': [], 'mouthLowerDownLeft': [], 'mouthLowerDownRight': [],
    'mouthPressLeft': [], 'mouthPressRight': [], 'mouthPucker': [], 'mouthRight': [], 'mouthRollLower': [],
    'mouthRollUpper': [], 'mouthShrugLower': [], 'mouthShrugUpper': [], 'mouthSmileLeft': [], 'mouthSmileRight': [],
    'mouthStretchLeft': [], 'mouthStretchRight': [], 'mouthUpperUpLeft': [], 'mouthUpperUpRight': [],
    'noseSneerLeft': [], 'noseSneerRight': []
}

# neutral_frames_path = sorted(glob.glob("frames_cleared/neutral/*"))
# disgust_frames_path = sorted(glob.glob("frames_cleared/disgust/*"))
# smile_frames_path = sorted(glob.glob("frames_cleared/smile/*"))

# neutral_frames_path = sorted(glob.glob("frames_filtered/neutral/*_2.png"))
# disgust_frames_path = sorted(glob.glob("frames_filtered/disgust/*_2.png"))
# smile_frames_path = sorted(glob.glob("frames_filtered/smile/*_2.png"))

neutral_frames_path = sorted(glob.glob("face_emo_datasets/frames_filtered/neutral/*_2.png"))
disgust_frames_path = sorted(glob.glob("face_emo_datasets/frames_filtered/disgust/*_2.png"))
smile_frames_path = sorted(glob.glob("face_emo_datasets/frames_filtered/smile/*_2.png"))

with FaceLandmarker.create_from_options(options) as landmarker:
    for frame_filename in neutral_frames_path:
        face_frame = cv.imread(frame_filename)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
        face_landmarker_result = landmarker.detect(mp_image)
        if len(face_landmarker_result.face_blendshapes) != 0:
            face_blendshapes_scores = get_face_blendshapes_scores(face_landmarker_result.face_blendshapes[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(face_blendshapes_scores[i])
            blendshapes_dict["filename"].append(frame_filename)
            blendshapes_dict["label"].append(0)

    for frame_filename in smile_frames_path:
        face_frame = cv.imread(frame_filename)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
        face_landmarker_result = landmarker.detect(mp_image)
        if len(face_landmarker_result.face_blendshapes) != 0:
            face_blendshapes_scores = get_face_blendshapes_scores(face_landmarker_result.face_blendshapes[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(face_blendshapes_scores[i])
            blendshapes_dict["filename"].append(frame_filename)
            blendshapes_dict["label"].append(1)

    for frame_filename in disgust_frames_path:
        face_frame = cv.imread(frame_filename)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
        face_landmarker_result = landmarker.detect(mp_image)
        if len(face_landmarker_result.face_blendshapes) != 0:
            face_blendshapes_scores = get_face_blendshapes_scores(face_landmarker_result.face_blendshapes[0])
            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(face_blendshapes_scores[i])
            blendshapes_dict["filename"].append(frame_filename)
            blendshapes_dict["label"].append(2)

blendshapes_df = pd.DataFrame.from_dict(blendshapes_dict)
print(blendshapes_df.head(5))
blendshapes_df.to_csv("frames_filtered_blendshapes.csv", index=False)



