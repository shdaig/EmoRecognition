import os
import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import warnings
warnings.filterwarnings('ignore')

steps_frames = 30

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../models/face_landmarker.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))
    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")
    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def get_face_blendshapes_scores(face_blendshapes):
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    return face_blendshapes_scores


if __name__ == "__main__":
    video_files_list = glob.glob("/home/neuron/mnt/a/A/arhiv/emoscroll/**/*.avi")

    for video_path in video_files_list:
        print(video_path)
        subj_dir = os.path.join("../blendshapes", video_path.split('/')[-2])
        os.mkdir(subj_dir)
        out_path = os.path.join(subj_dir, "blendshapes.csv")
        print(out_path)
        with FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)

            frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames_count)
            video_length_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            blendshapes_dict = {
                "time(s.)": [],
                '_neutral': [],
                'browDownLeft': [],
                'browDownRight': [],
                'browInnerUp': [],
                'browOuterUpLeft': [],
                'browOuterUpRight': [],
                'cheekPuff': [],
                'cheekSquintLeft': [],
                'cheekSquintRight': [],
                'eyeBlinkLeft': [],
                'eyeBlinkRight': [],
                'eyeLookDownLeft': [],
                'eyeLookDownRight': [],
                'eyeLookInLeft': [],
                'eyeLookInRight': [],
                'eyeLookOutLeft': [],
                'eyeLookOutRight': [],
                'eyeLookUpLeft': [],
                'eyeLookUpRight': [],
                'eyeSquintLeft': [],
                'eyeSquintRight': [],
                'eyeWideLeft': [],
                'eyeWideRight': [],
                'jawForward': [],
                'jawLeft': [],
                'jawOpen': [],
                'jawRight': [],
                'mouthClose': [],
                'mouthDimpleLeft': [],
                'mouthDimpleRight': [],
                'mouthFrownLeft': [],
                'mouthFrownRight': [],
                'mouthFunnel': [],
                'mouthLeft': [],
                'mouthLowerDownLeft': [],
                'mouthLowerDownRight': [],
                'mouthPressLeft': [],
                'mouthPressRight': [],
                'mouthPucker': [],
                'mouthRight': [],
                'mouthRollLower': [],
                'mouthRollUpper': [],
                'mouthShrugLower': [],
                'mouthShrugUpper': [],
                'mouthSmileLeft': [],
                'mouthSmileRight': [],
                'mouthStretchLeft': [],
                'mouthStretchRight': [],
                'mouthUpperUpLeft': [],
                'mouthUpperUpRight': [],
                'noseSneerLeft': [],
                'noseSneerRight': []
            }
            print(frames_count)
            face_blendshapes_names = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
                                      'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
                                      'eyeBlinkLeft',
                                      'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft',
                                      'eyeLookInRight',
                                      'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
                                      'eyeSquintLeft',
                                      'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
                                      'jawOpen',
                                      'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
                                      'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
                                      'mouthLowerDownRight',
                                      'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
                                      'mouthRollLower',
                                      'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
                                      'mouthSmileRight',
                                      'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
                                      'noseSneerLeft', 'noseSneerRight']
            prev_blendshapes_scores = [0.0 for j in range(52)]
            for frame_idx in range(0, frames_count, steps_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if not ret:
                    print('PASS')
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    face_landmarker_result = landmarker.detect(mp_image)
                    face_blendshapes_scores = prev_blendshapes_scores

                    if len(face_landmarker_result.face_blendshapes) != 0:
                        face_blendshapes_scores = get_face_blendshapes_scores(
                            face_landmarker_result.face_blendshapes[0])
                        prev_blendshapes_scores = face_blendshapes_scores

                    for i in range(len(face_blendshapes_names)):
                        blendshapes_dict[face_blendshapes_names[i]].append(face_blendshapes_scores[i])
                    blendshapes_dict["time(s.)"].append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

            blendshapes_df = pd.DataFrame.from_dict(blendshapes_dict)
            print(blendshapes_df.head(5))

            blendshapes_df.to_csv(out_path, index=False)
