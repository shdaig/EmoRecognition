import csv
import pandas as pd
import numpy as np
import glob


def calculate_distance(x, y, z):
    dist = np.sqrt(np.square(x[0] - x[1]) + np.square(y[0] - y[1]) + np.square(z[0] - z[1]))
    return dist


def calculate_distance_point(point1, point2):
    dist = np.sqrt(np.square(point1.x - point2.x) + np.square(point1.y - point2.y) + np.square(point1.z - point2.z))
    return dist


# points = {'nose_down':4,'nose_up':197}
def nose_dist_calc(data, points=None):
    if points is None:
        points = {'nose_down': 4, 'nose_up': 197}
    x_down, x_up = [], []
    y_down, y_up = [], []
    z_down, z_up = [], []
    for i in range(len(data)):
        point_down = points['nose_down']
        point_up = points['nose_up']
        x_down.append(data[i][point_down].x)
        y_down.append(data[i][point_down].y)
        z_down.append(data[i][point_down].z)
        x_up.append(data[i][point_up].x)
        y_up.append(data[i][point_up].y)
        z_up.append(data[i][point_up].z)
    nose_dist = []
    for i in range(len(x_up)):
        dist = calculate_distance(x = (x_down[i], x_up[i]), y = (y_down[i], y_up[i]), z = (z_down[i], z_up[i]))
        nose_dist.append(dist)
    return nose_dist


# points = {'nose_down':4, 'left_corner':291, 'right_corner':61}
def corner_dist_nose_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose_down': 4, 'left_corner': 291, 'right_corner': 61}
    x1, x2, x3 = [], [], []
    y1, y2, y3 = [], [], []
    z1, z2, z3 = [], [], []
    for i in range(len(data)):
        point1_left_corner = points['left_corner']
        point2_right_corner = points['right_corner']
        point3_down_nose = points['nose_down']
        x1.append(data[i][point1_left_corner].x)
        y1.append(data[i][point1_left_corner].y)
        z1.append(data[i][point1_left_corner].z)
        x2.append(data[i][point2_right_corner].x)
        y2.append(data[i][point2_right_corner].y)
        z2.append(data[i][point2_right_corner].z)
        x3.append(data[i][point3_down_nose].x)
        y3.append(data[i][point3_down_nose].y)
        z3.append(data[i][point3_down_nose].z)
    corner_nose_feate = []
    for i in range(len(x1)):
        dist1 = calculate_distance(x = (x1[i], x3[i]), y = (y1[i], y3[i]), z = (z1[i], z3[i]))
        dist2 = calculate_distance(x = (x2[i], x3[i]), y = (y2[i], y3[i]), z = (z2[i], z3[i]))
        dist1_norm = dist1 / nose_dist[i]
        dist2_norm = dist2 / nose_dist[i]
        dist_corners_nose = (dist1_norm + dist2_norm) / 2
        corner_nose_feate.append(dist_corners_nose)
    return corner_nose_feate


# points = {'left_corner':291, 'right_corner':61}
def corner_dist_calc(data, nose_dist, points=None):
    if points is None:
        points = {'left_corner': 291, 'right_corner': 61}
    x1, x2 = [], []
    y1, y2 = [], []
    z1, z2 = [], []
    for i in range(len(data)):
        point1_left_corner = points['left_corner']
        point2_right_corner = points['right_corner']
        x1.append(data[i][point1_left_corner].x)
        y1.append(data[i][point1_left_corner].y)
        z1.append(data[i][point1_left_corner].z)
        x2.append(data[i][point2_right_corner].x)
        y2.append(data[i][point2_right_corner].y)
        z2.append(data[i][point2_right_corner].z)
    corner_feate = []
    for i in range(len(x1)):
        dist_corner = calculate_distance(x = (x1[i], x2[i]), y = (y1[i], y2[i]), z = (z1[i], z2[i]))
        dist_corner_norm = dist_corner / nose_dist[i]
        corner_feate.append(dist_corner_norm)
    return corner_feate


# points = {'mouth_up_1': 37, 'mouth_up_2': 0, 'mouth_up_3': 267,
#           'mouth_down_1': 84, 'mouth_down_2': 17, 'mouth_down_3': 314}
def mouth_up_down_dist_calc(data, nose_dist, points=None):
    if points is None:
        points = {'mouth_up_1': 37, 'mouth_up_2': 0, 'mouth_up_3': 267, 'mouth_down_1': 84, 'mouth_down_2': 17,
                  'mouth_down_3': 314}
    x_up_1, x_up_2, x_up_3, x_down_1, x_down_2, x_down_3 = [], [], [], [], [], []
    y_up_1, y_up_2, y_up_3, y_down_1, y_down_2, y_down_3 = [], [], [], [], [], []
    z_up_1, z_up_2, z_up_3, z_down_1, z_down_2, z_down_3 = [], [], [], [], [], []
    for i in range(len(data)):
        point_up_1, point_up_2, point_up_3 = points['mouth_up_1'], points['mouth_up_2'], points['mouth_up_3']
        point_down_1, point_down_2, point_down_3 = points['mouth_down_1'], points['mouth_down_2'], points['mouth_down_3']
        x_up_1.append(data[i][point_up_1].x)
        y_up_1.append(data[i][point_up_1].y)
        z_up_1.append(data[i][point_up_1].z)
        x_up_2.append(data[i][point_up_2].x)
        y_up_2.append(data[i][point_up_2].y)
        z_up_2.append(data[i][point_up_2].z)
        x_up_3.append(data[i][point_up_3].x)
        y_up_3.append(data[i][point_up_3].y)
        z_up_3.append(data[i][point_up_3].z)
        x_down_1.append(data[i][point_down_1].x)
        y_down_1.append(data[i][point_down_1].y)
        z_down_1.append(data[i][point_down_1].z)
        x_down_2.append(data[i][point_down_2].x)
        y_down_2.append(data[i][point_down_2].y)
        z_down_2.append(data[i][point_down_2].z)
        x_down_3.append(data[i][point_down_3].x)
        y_down_3.append(data[i][point_down_3].y)
        z_down_3.append(data[i][point_down_3].z)
    mouth_up_down_feate = []
    for i in range(len(x_up_1)):
        dist1 = calculate_distance(x = (x_up_1[i], x_down_1[i]), y = (y_up_1[i], y_down_1[i]), z = (z_up_1[i], z_down_1[i])) / nose_dist[i]
        dist2 = calculate_distance(x = (x_up_2[i], x_down_2[i]), y = (y_up_2[i], y_down_2[i]), z = (z_up_2[i], z_down_2[i])) / nose_dist[i]
        dist3 = calculate_distance(x = (x_up_3[i], x_down_3[i]), y = (y_up_3[i], y_down_3[i]), z = (z_up_3[i], z_down_3[i])) / nose_dist[i]
        mouth_up_down_dist = (dist1+dist2+dist3) / 3
        mouth_up_down_feate.append(mouth_up_down_dist)
    return mouth_up_down_feate


# points = {'left_nose': 358, 'right_nose': 129, 'left_corner_eye': 362, 'right_corner_eye': 133}
def left_right_nose_calc(data, nose_dist, points=None):
    if points is None:
        points = {'left_nose': 358, 'right_nose': 129, 'left_corner_eye': 362, 'right_corner_eye': 133}
    x_left_nose, x_right_nose, x_left_eye, x_right_eye = [], [], [], []
    y_left_nose, y_right_nose, y_left_eye, y_right_eye = [], [], [], []
    z_left_nose, z_right_nose, z_left_eye, z_right_eye = [], [], [], []
    for i in range(len(data)):
        point_left_nose, point_right_nose = points['left_nose'], points['right_nose']
        point_left_eye, point_right_eye = points['left_corner_eye'], points['right_corner_eye']
        x_left_nose.append(data[i][point_left_nose].x)
        y_left_nose.append(data[i][point_left_nose].y)
        z_left_nose.append(data[i][point_left_nose].z)
        
        x_right_nose.append(data[i][point_right_nose].x)
        y_right_nose.append(data[i][point_right_nose].y)
        z_right_nose.append(data[i][point_right_nose].z)
        
        x_left_eye.append(data[i][point_left_eye].x)
        y_left_eye.append(data[i][point_left_eye].y)
        z_left_eye.append(data[i][point_left_eye].z)
        
        x_right_eye.append(data[i][point_right_eye].x)
        y_right_eye.append(data[i][point_right_eye].y)
        z_right_eye.append(data[i][point_right_eye].z)
    left_right_nose_feate = []
    for i in range(len(x_left_nose)):
        dist_left = calculate_distance(x = (x_left_nose[i], x_left_eye[i]), y = (y_left_nose[i], y_left_eye[i]), z = (z_left_nose[i], z_left_eye[i]))
        dist_right = calculate_distance(x = (x_right_nose[i], x_right_eye[i]), y = (y_right_nose[i], y_right_eye[i]), z = (z_right_nose[i], z_right_eye[i]))
        dist_left_norm = dist_left / nose_dist[i]
        dist_right_norm = dist_right / nose_dist[i]
        dist = (dist_left_norm + dist_right_norm) / 2
        left_right_nose_feate.append(dist)
    return left_right_nose_feate


# points = {'up_mouth': 0, 'down_nose': 1}
def up_mouth_calc(data, nose_dist, points=None):
    if points is None:
        points = {'up_mouth': 0, 'down_nose': 1}
    x_mouth, x_nose = [], []
    y_mouth, y_nose = [], []
    z_mouth, z_nose = [], []
    for i in range(len(data)):
        point_mouth = points['up_mouth']
        point_nose = points['down_nose']
        x_mouth.append(data[i][point_mouth].x)
        y_mouth.append(data[i][point_mouth].y)
        z_mouth.append(data[i][point_mouth].z)

        x_nose.append(data[i][point_nose].x)
        y_nose.append(data[i][point_nose].y)
        z_nose.append(data[i][point_nose].z)
    up_mouth_feate = []
    for i in range(len(x_mouth)):
        dist = calculate_distance(x = (x_mouth[i], x_nose[i]), y = (y_mouth[i], y_nose[i]), z = (z_mouth[i], z_nose[i]))
        dist_norm = dist / nose_dist[i]
        up_mouth_feate.append(dist_norm)
    return up_mouth_feate


# points = {'left_noses_fly':331, 'right_noses_fly':102}
def noses_fly_calc(data, nose_dist, points=None):
    if points is None:
        points = {'left_noses_fly': 331, 'right_noses_fly': 102}
    x_left, x_right = [], []
    y_left, y_right = [], []
    z_left, z_right = [], []
    # point_left = []
    # point_right = []
    for i in range(len(data)):
        point_left = points['left_noses_fly']
        point_right = points['right_noses_fly']
        x_left.append(data[i][point_left].x)
        y_left.append(data[i][point_left].y)
        z_left.append(data[i][point_left].z)
        # point_left.append(data[i][point_left_idx])

        x_right.append(data[i][point_right].x)
        y_right.append(data[i][point_right].y)
        z_right.append(data[i][point_right].z)
    noses_fly_feate = []
    for i in range(len(x_left)):
        dist = calculate_distance(x = (x_left[i], x_right[i]), y = (y_left[i], y_right[i]), z = (z_left[i], z_right[i]))
        dist_norm = dist / nose_dist[i]
        noses_fly_feate.append(dist_norm)
    return noses_fly_feate


# points = {'left_up_1':386, 'left_up_2':385, 'left_down_1':374, 'left_down_2':380, 
#           'right_up_1':159, 'right_up_2':158, 'right_down_1':145, 'right_down_2':153}
def eye_calc(data, nose_dist, points=None):
    if points is None:
        points = {'left_up_1': 386, 'left_up_2': 385, 'left_down_1': 374, 'left_down_2': 380, 'right_up_1': 159,
                  'right_up_2': 158, 'right_down_1': 145, 'right_down_2': 153}
    point_up_left_1, point_up_left_2 = [], []
    point_down_left_1, point_down_left_2 = [], []
    point_up_right_1, point_up_right_2 = [], []
    point_down_right_1, point_down_right_2 = [], []
    for i in range(len(data)):
        point_up_left_1_idx, point_up_left_2_idx = points['left_up_1'], points['left_up_2']
        point_down_left_1_idx, point_down_left_2_idx = points['left_down_1'], points['left_down_2']
        point_up_right_1_idx, point_up_right_2_idx = points['right_up_1'], points['right_up_2']
        point_down_right_1_idx, point_down_right_2_idx = points['right_down_1'], points['right_down_2']
        
        point_up_left_1.append(data[i][point_up_left_1_idx])
        point_up_left_2.append(data[i][point_up_left_2_idx])
        point_down_left_1.append(data[i][point_down_left_1_idx])
        point_down_left_2.append(data[i][point_down_left_2_idx])

        point_up_right_1.append(data[i][point_up_right_1_idx])
        point_up_right_2.append(data[i][point_up_right_2_idx])
        point_down_right_1.append(data[i][point_down_right_1_idx])
        point_down_right_2.append(data[i][point_down_right_2_idx])
    eye_feate = []
    for i in range(len(point_up_left_1)):
        dest_left_1 = calculate_distance_point(point1 = point_up_left_1[i], point2 = point_down_left_1[i]) / nose_dist[i]
        dest_left_2 = calculate_distance_point(point1 = point_up_left_2[i], point2 = point_down_left_2[i]) / nose_dist[i]
        dist_left = dest_left_1 + dest_left_2

        dest_right_1 = calculate_distance_point(point1 = point_up_right_1[i], point2 = point_down_right_1[i]) / nose_dist[i]
        dest_right_2 = calculate_distance_point(point1 = point_up_right_2[i], point2 = point_down_right_2[i]) / nose_dist[i]
        dist_right = dest_right_1 + dest_right_2

        dist = (dist_left + dist_right) / 4
        eye_feate.append(dist)
    return eye_feate


# points = {'nose_down':4, 'left_brov_1':285, 'left_brov_2':295, 'left_brov_3':282,
#           'right_brov_1':55, 'right_brov_2':65, 'right_brov_3':52}
def brov_calc(data, nose_dist, points=None):
    if points is None:
        points = {'nose_down': 4, 'left_brov_1': 285, 'left_brov_2': 295, 'left_brov_3': 282, 'right_brov_1': 55,
                  'right_brov_2': 65, 'right_brov_3': 52}
    point_nose = []
    point_left_brov_1, point_left_brov_2, point_left_brov_3 = [], [], []
    point_right_brov_1, point_right_brov_2, point_right_brov_3 = [], [], []
    for i in range(len(data)):
        point_nose_idx = points['nose_down']
        point_left_brov_1_idx, point_left_brov_2_idx, point_left_brov_3_idx = points['left_brov_1'], points['left_brov_2'], points['left_brov_3']
        point_right_brov_1_idx, point_right_brov_2_idx, point_right_brov_3_idx = points['right_brov_1'], points['right_brov_2'], points['right_brov_3']
        point_nose.append(data[i][point_nose_idx])

        point_left_brov_1.append(data[i][point_left_brov_1_idx])
        point_left_brov_2.append(data[i][point_left_brov_2_idx])
        point_left_brov_3.append(data[i][point_left_brov_3_idx])

        point_right_brov_1.append(data[i][point_right_brov_1_idx])
        point_right_brov_2.append(data[i][point_right_brov_2_idx])
        point_right_brov_3.append(data[i][point_right_brov_3_idx])
    brov_feate = []
    for i in range(len(point_nose)):
        dist_left_1 = calculate_distance_point(point1 = point_left_brov_1[i], point2 = point_nose[i]) / nose_dist[i]
        dist_left_2 = calculate_distance_point(point1 = point_left_brov_2[i], point2 = point_nose[i]) / nose_dist[i]
        dist_left_3 = calculate_distance_point(point1 = point_left_brov_3[i], point2 = point_nose[i]) / nose_dist[i]
        dist_left = (dist_left_1 + dist_left_2 + dist_left_3) / 3

        dist_right_1 = calculate_distance_point(point1 = point_right_brov_1[i], point2 = point_nose[i]) / nose_dist[i]
        dist_right_2 = calculate_distance_point(point1 = point_right_brov_2[i], point2 = point_nose[i]) / nose_dist[i]
        dist_right_3 = calculate_distance_point(point1 = point_right_brov_3[i], point2 = point_nose[i]) / nose_dist[i]
        dist_right = (dist_right_1 + dist_right_2 + dist_right_3) / 3

        dist = dist_left + dist_right
        brov_feate.append(dist)
    return brov_feate


# point_mouth_updown_downup = {'updown':14, 'downup':17}
def new_feat_mouth_updown(data, nose_dist, points=None):
    if points is None:
        points = {'updown': 14, 'downup': 17}
    x_down, x_up = [], []
    y_down, y_up = [], []
    z_down, z_up = [], []
    for i in range(len(data)):
        up_down_mouth = points['updown']
        down_down_mouth = points['downup']
        x_down.append(data[i][down_down_mouth].x)
        y_down.append(data[i][down_down_mouth].y)
        z_down.append(data[i][down_down_mouth].z)
        x_up.append(data[i][up_down_mouth].x)
        y_up.append(data[i][up_down_mouth].y)
        z_up.append(data[i][up_down_mouth].z)
    down_mouth = []
    for i in range(len(x_up)):
        dist = calculate_distance(x = (x_down[i], x_up[i]), y = (y_down[i], y_up[i]), z = (z_down[i], z_up[i])) / nose_dist[i]
        down_mouth.append(dist)
    return down_mouth