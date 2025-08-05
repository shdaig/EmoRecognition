import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

steps_frames = 30
# in_file = '20240409_am39/cam1.20240409T131819745852.avi'
in_file = '../local_data/20240409_am35/cam1.20240409T110754277664.avi'
out_file = 'am35.csv'


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
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


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../face_landmarker.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(in_file)

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

    prev_blendshapes_scores = [0.0 for j in range(52)]

    for frame_idx in range(0, frames_count, steps_frames):
    # for frame_idx in range(0, 120, 30):
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
                face_blendshapes_scores = get_face_blendshapes_scores(face_landmarker_result.face_blendshapes[0])
                prev_blendshapes_scores = face_blendshapes_scores

            for i in range(len(face_blendshapes_names)):
                blendshapes_dict[face_blendshapes_names[i]].append(face_blendshapes_scores[i])
            blendshapes_dict["time(s.)"].append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

    blendshapes_df = pd.DataFrame.from_dict(blendshapes_dict)
    print(blendshapes_df.head(5))

    blendshapes_df.to_csv(out_file, index=False)
