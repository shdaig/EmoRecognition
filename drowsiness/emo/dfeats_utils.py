import numpy as np

nose_center = 94  # 1


def calculate_distance(x, y, z):
    dist = np.sqrt(np.square(x[0] - x[1]) + np.square(y[0] - y[1]) + np.square(z[0] - z[1]))
    return dist


def calculate_distance_point(point1, point2):
    dist = np.sqrt(np.square(point1.x - point2.x) + np.square(point1.y - point2.y) + np.square(point1.z - point2.z))
    return dist


def nose_dist_calc(data, points=None):
    if points is None:
        points = {'nose_down': 4, 'nose_up': 197}
    point_down = points['nose_down']
    point_up = points['nose_up']
    return calculate_distance_point(data[point_down], data[point_up])


def center_right_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'center': 13, 'right': 291}
    point1 = points['center']
    point2 = points['right']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def center_left_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'center': 13, 'left': 61}
    point1 = points['center']
    point2 = points['left']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_right_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'right': 291}
    point1 = points['nose']
    point2 = points['right']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_left_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'left': 61}
    point1 = points['nose']
    point2 = points['left']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def horizontal_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'right': 291, 'left': 61}
    point1 = points['right']
    point2 = points['left']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def vertical_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'up': 0, 'down': 17}
    point1 = points['up']
    point2 = points['down']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_upper_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'upper_lip': 0}
    point1 = points['nose']
    point2 = points['upper_lip']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_lower_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'lower_lip': 17}
    point1 = points['nose']
    point2 = points['lower_lip']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_left_upper_lip_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'left_upper_lip': 39}
    point1 = points['nose']
    point2 = points['left_upper_lip']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_right_upper_lip_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'right_upper_lip': 269}
    point1 = points['nose']
    point2 = points['right_upper_lip']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_wings_calc(data, nose_dist, points=None):
    if points is None:
        points = {'left_wing': 48, 'right_wing': 278}
    point1 = points['left_wing']
    point2 = points['right_wing']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def inner_brows_calc(data, nose_dist, points=None):
    if points is None:
        points = {'left_brow': 66, 'right_brow': 296}
    point1 = points['left_brow']
    point2 = points['right_brow']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_left_brow_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'left_brow': 222}
    point1 = points['left_brow']
    point2 = points['nose']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_right_brow_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'right_brow': 442}
    point1 = points['nose']
    point2 = points['right_brow']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def left_cheek_eye(data, nose_dist, points=None):
    if points is None:
        points = {'left_eye': 133, 'left_cheek': 101}
    point1 = points['left_eye']
    point2 = points['left_cheek']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def right_cheek_eye(data, nose_dist, points=None):
    if points is None:
        points = {'right_eye': 362, 'right_cheek': 330}
    point1 = points['right_eye']
    point2 = points['right_cheek']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def nose_chin(data, nose_dist, points=None):
    if points is None:
        points = {'nose': nose_center, 'chin': 199}
    point1 = points['nose']
    point2 = points['chin']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def forehead_left_brow_in(data, nose_dist, points=None):
    if points is None:
        points = {'forehead': 109, 'brow_in': 107}
    point1 = points['forehead']
    point2 = points['brow_in']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def forehead_right_brow_in(data, nose_dist, points=None):
    if points is None:
        points = {'forehead': 338, 'brow_in': 336}
    point1 = points['forehead']
    point2 = points['brow_in']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def forehead_left_brow_c(data, nose_dist, points=None):
    if points is None:
        points = {'forehead': 103, 'brow_c': 105}
    point1 = points['forehead']
    point2 = points['brow_c']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def forehead_right_brow_c(data, nose_dist, points=None):
    if points is None:
        points = {'forehead': 332, 'brow_c': 334}
    point1 = points['forehead']
    point2 = points['brow_c']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def forehead_left_brow_out(data, nose_dist, points=None):
    if points is None:
        points = {'forehead': 21, 'brow_out': 70}
    point1 = points['forehead']
    point2 = points['brow_out']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def forehead_right_brow_out(data, nose_dist, points=None):
    if points is None:
        points = {'forehead': 251, 'brow_out': 300}
    point1 = points['forehead']
    point2 = points['brow_out']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def left_cheekbone_upper_lip(data, nose_dist, points=None):
    if points is None:
        points = {'left_cheekbone': 352, 'left_upper_lip': 269}
    point1 = points['left_cheekbone']
    point2 = points['left_upper_lip']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def right_cheekbone_upper_lip(data, nose_dist, points=None):
    if points is None:
        points = {'right_cheekbone': 123, 'right_upper_lip': 39}
    point1 = points['right_cheekbone']
    point2 = points['right_upper_lip']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def left_mouth_out(data, nose_dist, points=None):
    if points is None:
        points = {'mouth': 291, 'chin': 434}
    point1 = points['mouth']
    point2 = points['chin']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def right_mouth_out(data, nose_dist, points=None):
    if points is None:
        points = {'mouth': 61, 'chin': 214}
    point1 = points['mouth']
    point2 = points['chin']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def left_chin_lower_lip(data, nose_dist, points=None):
    if points is None:
        points = {'mouth': 405, 'chin': 262}
    point1 = points['mouth']
    point2 = points['chin']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def right_chin_lower_lip(data, nose_dist, points=None):
    if points is None:
        points = {'mouth': 181, 'chin': 32}
    point1 = points['mouth']
    point2 = points['chin']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def centered_left_chin_lower_lip(data, nose_dist, points=None):
    if points is None:
        points = {'mouth': 314, 'chin': 428}
    point1 = points['mouth']
    point2 = points['chin']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def centered_right_chin_lower_lip(data, nose_dist, points=None):
    if points is None:
        points = {'mouth': 89, 'chin': 208}
    point1 = points['mouth']
    point2 = points['chin']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def left_upper_eye_brow(data, nose_dist, points=None):
    if points is None:
        points = {'eye': 259, 'brow': 276}
    point1 = points['eye']
    point2 = points['brow']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def right_upper_eye_brow(data, nose_dist, points=None):
    if points is None:
        points = {'eye': 29, 'brow': 46}
    point1 = points['eye']
    point2 = points['brow']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def left_eye_brow(data, nose_dist, points=None):
    if points is None:
        points = {'eye': 359, 'brow': 276}
    point1 = points['eye']
    point2 = points['brow']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def right_eye_brow(data, nose_dist, points=None):
    if points is None:
        points = {'eye': 130, 'brow': 46}
    point1 = points['eye']
    point2 = points['brow']
    return calculate_distance_point(data[point1], data[point2]) / nose_dist


def get_face_landmarks_features(face_landmarks):
    data = face_landmarks

    nose_dist = nose_dist_calc(data)
    center_right_mouth_calc_v = center_right_mouth_calc(data, nose_dist)
    center_left_mouth_calc_v = center_left_mouth_calc(data, nose_dist)
    nose_right_mouth_calc_v = nose_right_mouth_calc(data, nose_dist)
    nose_left_mouth_calc_v = nose_left_mouth_calc(data, nose_dist)
    horizontal_mouth_calc_v = horizontal_mouth_calc(data, nose_dist)
    vertical_mouth_calc_v = vertical_mouth_calc(data, nose_dist)
    nose_upper_mouth_calc_v = nose_upper_mouth_calc(data, nose_dist)
    nose_lower_mouth_calc_v = nose_lower_mouth_calc(data, nose_dist)
    nose_left_upper_lip_calc_v = nose_left_upper_lip_calc(data, nose_dist)
    nose_right_upper_lip_calc_v = nose_right_upper_lip_calc(data, nose_dist)
    nose_wings_calc_v = nose_wings_calc(data, nose_dist)
    inner_brows_calc_v = inner_brows_calc(data, nose_dist)
    nose_left_brow_calc_v = nose_left_brow_calc(data, nose_dist)
    nose_right_brow_calc_v = nose_right_brow_calc(data, nose_dist)
    left_cheek_eye_v = left_cheek_eye(data, nose_dist)
    right_cheek_eye_v = right_cheek_eye(data, nose_dist)
    nose_chin_v = nose_chin(data, nose_dist)
    forehead_left_brow_in_v = forehead_left_brow_in(data, nose_dist)
    forehead_right_brow_in_v = forehead_right_brow_in(data, nose_dist)
    forehead_left_brow_c_v = forehead_left_brow_c(data, nose_dist)
    forehead_right_brow_c_v = forehead_right_brow_c(data, nose_dist)
    forehead_left_brow_out_v = forehead_left_brow_out(data, nose_dist)
    forehead_right_brow_out_v = forehead_right_brow_out(data, nose_dist)
    left_cheekbone_upper_lip_v = left_cheekbone_upper_lip(data, nose_dist)
    right_cheekbone_upper_lip_v = right_cheekbone_upper_lip(data, nose_dist)
    left_mouth_out_v = left_mouth_out(data, nose_dist)
    right_mouth_out_v = right_mouth_out(data, nose_dist)
    left_chin_lower_lip_v = left_chin_lower_lip(data, nose_dist)
    right_chin_lower_lip_v = right_chin_lower_lip(data, nose_dist)
    centered_left_chin_lower_lip_v = centered_left_chin_lower_lip(data, nose_dist)
    centered_right_chin_lower_lip_v = centered_right_chin_lower_lip(data, nose_dist)
    left_upper_eye_brow_v = left_upper_eye_brow(data, nose_dist)
    right_upper_eye_brow_v = right_upper_eye_brow(data, nose_dist)
    left_eye_brow_v = left_eye_brow(data, nose_dist)
    right_eye_brow_v = right_eye_brow(data, nose_dist)

    face_blendshapes_scores = [center_right_mouth_calc_v, center_left_mouth_calc_v,
                               nose_right_mouth_calc_v, nose_left_mouth_calc_v,
                               horizontal_mouth_calc_v, vertical_mouth_calc_v,
                               nose_upper_mouth_calc_v, nose_lower_mouth_calc_v,
                               nose_left_upper_lip_calc_v, nose_right_upper_lip_calc_v,
                               nose_wings_calc_v, inner_brows_calc_v,
                               nose_left_brow_calc_v, nose_right_brow_calc_v,
                               left_cheek_eye_v, right_cheek_eye_v, nose_chin_v,
                               forehead_left_brow_in_v, forehead_right_brow_in_v,
                               forehead_left_brow_c_v, forehead_right_brow_c_v,
                               forehead_left_brow_out_v, forehead_right_brow_out_v,
                               left_cheekbone_upper_lip_v, right_cheekbone_upper_lip_v,
                               left_mouth_out_v, right_mouth_out_v,
                               left_chin_lower_lip_v, right_chin_lower_lip_v,
                               centered_left_chin_lower_lip_v, centered_right_chin_lower_lip_v,
                               left_upper_eye_brow_v, right_upper_eye_brow_v,
                               left_eye_brow_v, right_eye_brow_v]

    return face_blendshapes_scores


def get_features_names():
    names = ['center_right_mouth_calc', 'center_left_mouth_calc',
             'nose_right_mouth_calc', 'nose_left_mouth_calc',
             'horizontal_mouth_calc', 'vertical_mouth_calc',
             'nose_upper_mouth_calc', 'nose_lower_mouth_calc',
             'nose_left_upper_lip_calc', 'nose_right_upper_lip_calc',
             'nose_wings_calc', 'inner_brows_calc',
             'nose_left_brow_calc', 'nose_right_brow_calc',
             'left_cheek_eye', 'right_cheek_eye', 'nose_chin',
             'forehead_left_brow_in', 'forehead_right_brow_in', 'forehead_left_brow_c',
             'forehead_right_brow_c', 'forehead_left_brow_out', 'forehead_right_brow_out',
             'left_cheekbone_upper_lip', 'right_cheekbone_upper_lip', 'left_mouth_out',
             'right_mouth_out', 'left_chin_lower_lip', 'right_chin_lower_lip',
             'centered_left_chin_lower_lip', 'centered_right_chin_lower_lip',
             'left_upper_eye_brow', 'right_upper_eye_brow', 'left_eye_brow', 'right_eye_brow']

    return names


def get_features_dict():
    names = get_features_names()
    result_dict = {"filename": [], "label": []}
    for name in names:
        result_dict[name] = []
    return result_dict
