import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import warnings
warnings.filterwarnings("ignore")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../../models/face_landmarker.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)

cap = cv2.VideoCapture(0)


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
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        face_landmarker_result = landmarker.detect(mp_image)
        image = draw_landmarks_on_image(image, face_landmarker_result)
        img_h, img_w, img_c = image.shape
        face_2d_points = []

        face_3d_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -9.5, -4.0),  # Chin
            (5.5, 7.5, -5.0),  # Left eye left corner
            (-5.5, 7.5, -5.0),  # Right eye right corner
            (6.6, -3.0, -7.0),  # Left Mouth corner
            (-6.6, -3.0, -7.0),  # Right mouth corner
        ])

        if len(face_landmarker_result.face_landmarks) != 0:
            landmarks = face_landmarker_result.face_landmarks[0]
            nose_2d = (landmarks[1].x * img_w, landmarks[1].y * img_h)

            face_2d_points = np.array([
                (landmarks[1].x * img_w, landmarks[1].y * img_h),  # Nose tip
                (landmarks[199].x * img_w, landmarks[199].y * img_h),  # Chin
                (landmarks[33].x * img_w, landmarks[33].y * img_h),  # Left eye left corner
                (landmarks[263].x * img_w, landmarks[263].y * img_h),  # Right eye right corne
                (landmarks[192].x * img_w, landmarks[192].y * img_h),  # Left Mouth corner
                (landmarks[416].x * img_w, landmarks[416].y * img_h)  # Right mouth corner
            ], dtype=np.float64)

            # face_3d_points = np.array([
            #     (0.0, 0.0, 0.0),  # Nose tip
            #     (landmarks[1].x - landmarks[199].x,
            #      landmarks[1].y - landmarks[199].y,
            #      landmarks[1].z - landmarks[199].z),  # Chin
            #     (landmarks[1].x - landmarks[33].x,
            #      landmarks[1].y - landmarks[33].y,
            #      landmarks[1].z - landmarks[33].z),  # Left eye left corner
            #     (landmarks[1].x - landmarks[263].x,
            #      landmarks[1].y - landmarks[263].y,
            #      landmarks[1].z - landmarks[263].z),  # Right eye right corne
            #     (landmarks[1].x - landmarks[192].x,
            #      landmarks[1].y - landmarks[192].y,
            #      landmarks[1].z - landmarks[192].z),  # Left Mouth corner
            #     (landmarks[1].x - landmarks[416].x,
            #      landmarks[1].y - landmarks[416].y,
            #      landmarks[1].z - landmarks[416].z)  # Right mouth corner
            # ], dtype=np.float64)
            #
            # face_3d_points *= 100
            # print(face_3d_points)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]], dtype=np.float64)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            _, rot_vec, trans_vec = cv2.solvePnP(face_3d_points, face_2d_points, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0]
            y = angles[1]
            z = angles[2]

            print(x, y, z)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] - y), int(nose_2d[1] + x))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

        cv2.imshow('live cam', image)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
