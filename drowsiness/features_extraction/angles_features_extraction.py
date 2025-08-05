import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def _angle_from_vertical(pitch, roll, yaw):
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)
    vertical_angles = []
    for i in range(pitch_rad.shape[0]):
        rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch_rad[i]), -np.sin(pitch_rad[i])],
                       [0, np.sin(pitch_rad[i]), np.cos(pitch_rad[i])]])

        ry = np.array([[np.cos(roll_rad[i]), 0, np.sin(roll_rad[i])],
                       [0, 1, 0],
                       [-np.sin(roll_rad[i]), 0, np.cos(roll_rad[i])]])

        rz = np.array([[np.cos(yaw_rad[i]), -np.sin(yaw_rad[i]), 0],
                       [np.sin(yaw_rad[i]), np.cos(yaw_rad[i]), 0],
                       [0, 0, 1]])

        r = rx @ ry @ rz

        oy = np.array([0, 1, 0])
        angled_vector = r @ oy
        dot_product = angled_vector[1]
        norm_vector = np.sqrt(angled_vector[0]**2 + angled_vector[1]**2 + angled_vector[2]**2)
        angle_rad = np.arccos(dot_product / norm_vector)
        angle_deg = np.degrees(angle_rad)
        vertical_angles.append(angle_deg)
    vertical_angles = np.array(vertical_angles)
    return vertical_angles


def head_angles_features_counts(angles_data_fname: str, window: int, shift: int, q_discrete, x_state):
    """
    Returns array of counts for x-axis and named arrays of features in dictionary
    :param angles_data_fname: CSV file with data [timestamp of frame, pitch, roll, yaw]
    :param window: window for features calculation
    :param shift: shift of window
    :return: counts for x-axis, dictionary with features array
    """
    df_angles = pd.read_csv(angles_data_fname)
    df_angles = df_angles.fillna(-30.)

    timestamps_raw = df_angles["timestamp"].to_numpy() / 1000
    pitch_raw = df_angles["pitch"].to_numpy()
    roll_raw = df_angles["roll"].to_numpy()
    yaw_raw = df_angles["yaw"].to_numpy()

    work_start_idx = np.argwhere(q_discrete == 1)[0][0]
    work_start_time = x_state[work_start_idx]
    start_x = np.argwhere(timestamps_raw >= work_start_time)[0][0]
    pitch_norm = np.mean(pitch_raw[start_x:start_x + 1800])
    roll_norm = np.mean(roll_raw[start_x:start_x + 1800])
    yaw_norm = np.mean(yaw_raw[start_x:start_x + 1800])
    pitch_raw -= pitch_norm
    roll_raw -= roll_norm
    yaw_raw -= yaw_norm

    pitch_raw_abs = np.abs(pitch_raw)
    roll_raw_abs = np.abs(roll_raw)
    yaw_raw_abs = np.abs(yaw_raw)

    end_time = timestamps_raw[-1]
    segment_end_time = window
    
    pitch_mean = []
    roll_mean = []
    yaw_mean = []
    pitch_abs_mean = []
    roll_abs_mean = []
    yaw_abs_mean = []
    while segment_end_time <= end_time + shift:
        pitch_mean.append(np.mean(pitch_raw[np.all([timestamps_raw >= segment_end_time - window,
                                                    timestamps_raw <= segment_end_time], axis=0)]))
        roll_mean.append(np.mean(roll_raw[np.all([timestamps_raw >= segment_end_time - window,
                                                 timestamps_raw <= segment_end_time], axis=0)]))
        yaw_mean.append(np.mean(yaw_raw[np.all([timestamps_raw >= segment_end_time - window,
                                                timestamps_raw <= segment_end_time], axis=0)]))
        pitch_abs_mean.append(np.mean(pitch_raw_abs[np.all([timestamps_raw >= segment_end_time - window,
                                                    timestamps_raw <= segment_end_time], axis=0)]))
        roll_abs_mean.append(np.mean(roll_raw_abs[np.all([timestamps_raw >= segment_end_time - window,
                                                 timestamps_raw <= segment_end_time], axis=0)]))
        yaw_abs_mean.append(np.mean(yaw_raw_abs[np.all([timestamps_raw >= segment_end_time - window,
                                                timestamps_raw <= segment_end_time], axis=0)]))
        segment_end_time += shift
    pitch_mean = np.array(pitch_mean)
    roll_mean = np.array(roll_mean)
    yaw_mean = np.array(yaw_mean)
    pitch_abs_mean = np.array(pitch_abs_mean)
    roll_abs_mean = np.array(roll_abs_mean)
    yaw_abs_mean = np.array(yaw_abs_mean)
    max_abs_mean = np.max(np.vstack((pitch_abs_mean, roll_abs_mean, yaw_abs_mean)), axis=0)
    vertical_angle = _angle_from_vertical(pitch_mean, roll_mean, yaw_mean)
    
    x_counts = np.array(list(range(window, window + pitch_mean.shape[0] * shift, shift)))
    features = {
        "pitch_mean": pitch_mean,
        "roll_mean": roll_mean,
        "yaw_mean": yaw_mean,
        "pitch_abs_mean": pitch_abs_mean,
        "roll_abs_mean": roll_abs_mean,
        "yaw_abs_mean": yaw_abs_mean,
        "max_abs_mean": max_abs_mean,
        "vertical_angle": vertical_angle
    }

    return x_counts, features
