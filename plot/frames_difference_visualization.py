import glob

import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../models/face_landmarker.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)


def draw_landmarks(img, landmarks):
    img_h = img.shape[0]
    img_w = img.shape[1]

    annotated_img = np.copy(img)
    for landmark in landmarks:
        cv.circle(annotated_img, (int(landmark.x * img_w), int(landmark.y * img_h)), 1,
                  color=(75, 138, 201), thickness=-1)

    return annotated_img


def draw_diff_lines(img, landmarks1, landmarks2):
    img_h = img.shape[0]
    img_w = img.shape[1]

    annotated_img = np.copy(img)
    for i in range(len(landmarks1)):
        cv.line(annotated_img,
                (int(landmarks1[i].x * img_w), int(landmarks1[i].y * img_h)),
                (int(landmarks2[i].x * img_w), int(landmarks2[i].y * img_h)),
                color=(255, 0, 0), thickness=1)

    return annotated_img


# frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/neutral/*_1.png"))
frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/disgust/*_1.png"))
# frames_path_1 = sorted(glob.glob("face_emo_datasets/frames_filtered/smile/*_1.png"))

with FaceLandmarker.create_from_options(options) as landmarker:
    for frame_1_path in frames_path_1:
        frame_2_path = frame_1_path[:-5] + '2.png'

        frame_1 = cv.imread(frame_1_path)
        frame_2 = cv.imread(frame_2_path)
        frame_1 = cv.cvtColor(frame_1, cv.COLOR_BGR2RGB)
        frame_2 = cv.cvtColor(frame_2, cv.COLOR_BGR2RGB)
        mp_image_1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_1)
        mp_image_2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_2)
        face_landmarker_result_1 = landmarker.detect(mp_image_1)
        face_landmarker_result_2 = landmarker.detect(mp_image_2)

        if len(face_landmarker_result_1.face_landmarks) != 0 and len(face_landmarker_result_2.face_landmarks) != 0:
            face_1_landmarks = face_landmarker_result_1.face_landmarks[0]
            face_2_landmarks = face_landmarker_result_2.face_landmarks[0]

            annotated_frame_1 = draw_landmarks(frame_1, face_1_landmarks)
            annotated_frame_2 = draw_landmarks(frame_2, face_2_landmarks)
            annotated_frame_2 = draw_diff_lines(annotated_frame_2, face_1_landmarks, face_2_landmarks)

            fig = plt.figure(figsize=(15, 10))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222, projection='3d')
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224, projection='3d')

            xs_1 = []
            ys_1 = []
            zs_1 = []
            for landmark in face_1_landmarks:
                xs_1.append(landmark.x)
                ys_1.append(landmark.y)
                zs_1.append(landmark.z)
            ax2.scatter(xs_1, ys_1, zs_1, marker='o')
            highlight_point = 262
            ax2.scatter([face_1_landmarks[highlight_point].x],
                        [face_1_landmarks[highlight_point].y],
                        [face_1_landmarks[highlight_point].z], marker='o')
            ax2.view_init(elev=-90., azim=-90.)

            xs_2 = []
            ys_2 = []
            zs_2 = []
            for landmark in face_2_landmarks:
                xs_2.append(landmark.x)
                ys_2.append(landmark.y)
                zs_2.append(landmark.z)
            ax4.scatter(xs_2, ys_2, zs_2, marker='o')
            ax4.view_init(elev=-90., azim=-90.)

            ax1.imshow(annotated_frame_1)
            ax3.imshow(annotated_frame_2)
            fig.tight_layout()
            plt.show()
