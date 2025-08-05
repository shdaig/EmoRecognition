import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

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
x_axis = list(range(52))

with FaceLandmarker.create_from_options(options) as landmarker:

    fig = plt.figure()
    cap = cv2.VideoCapture(0)

    while True:
        prev_blendshapes_scores = [0.0 for j in range(52)]

        # display camera feed
        ret, frame = cap.read()

        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect(mp_image)
        face_blendshapes_scores = prev_blendshapes_scores

        if len(face_landmarker_result.face_blendshapes) != 0:
            face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_landmarker_result.face_blendshapes[0]]

        plt.cla()
        plt.ylim(0, 1)
        plt.bar(x_axis, face_blendshapes_scores)
        plt.xticks(x_axis, face_blendshapes_names, rotation=90)
        plt.grid(True)
        plt.tight_layout()

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        cv2.imshow("plot", img)
        cv2.imshow("cam", frame)

        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break
