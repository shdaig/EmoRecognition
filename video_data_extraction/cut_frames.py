import os
import glob
import shutil

import cv2 as cv
import numpy as np
import pandas as pd


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == "__main__":
    frame_subjs_idxs = {}

    labels_paths = glob.glob("/home/neuron/mnt/a/A/arhiv/emoscroll/*/face_labels.csv")
    labels_paths.remove("/home/neuron/mnt/a/A/arhiv/emoscroll/20240411_am54/face_labels.csv")

    for labels_path in labels_paths:
        labels_splited = labels_path.split('/')
        frame_idx_path = os.path.join("..", "frame_idx", labels_splited[-2], "frame_idx.csv")
        frame_subjs_idxs[labels_splited[-2]] = {"labels": [],
                                                "frames": []}
        try:
            face_labels_df = pd.read_csv(labels_path, header=None, dtype=str)
            frame_idx_df = pd.read_csv(frame_idx_path)
            timestamps_seconds = []
            labels = []

            for index, row in face_labels_df.iterrows():
                timest = row[0]
                timest_seconds = int(timest.split('.')[0]) * 60 + int(timest.split('.')[1])
                timestamps_seconds.append(timest_seconds)
                label = int(row[1])
                label = -1 if label == -2 else label
                label = 1 if label == 2 else label
                labels.append(label)
            frame_idx_timestamps = frame_idx_df["time(s.)"].to_numpy()
            frame_idx_features = frame_idx_df.drop("time(s.)", axis=1)

            prev_timest_seconds = 0

            for idx, timest in enumerate(timestamps_seconds):
                df_idx = find_nearest_idx(frame_idx_timestamps, timest)
                nearest_frame_idx = int(frame_idx_features.iloc[df_idx].loc[["frame_idx"]][0])

                if abs(prev_timest_seconds - timest) > 1.5:
                    frame_subjs_idxs[labels_splited[-2]]["frames"].append(nearest_frame_idx - 60 * 5)
                    frame_subjs_idxs[labels_splited[-2]]["labels"].append(0)
                prev_timest_seconds = timest

                frame_subjs_idxs[labels_splited[-2]]["frames"].append(nearest_frame_idx)
                frame_subjs_idxs[labels_splited[-2]]["labels"].append(labels[idx])
        except Exception:
            pass

    frames_save_dir = os.path.join("..", "frames_pairs")
    if os.path.exists(frames_save_dir):
        shutil.rmtree(frames_save_dir)
    os.mkdir(frames_save_dir)
    neutral_frames_dir = os.path.join(frames_save_dir, "neutral")
    disgust_frames_dir = os.path.join(frames_save_dir, "disgust")
    smile_frames_dir = os.path.join(frames_save_dir, "smile")
    os.mkdir(neutral_frames_dir)
    os.mkdir(disgust_frames_dir)
    os.mkdir(smile_frames_dir)

    for subj in frame_subjs_idxs:
        print(subj)
        video_file = glob.glob(f"/home/neuron/mnt/a/A/arhiv/emoscroll/{subj}/*.avi")[0]
        frames = frame_subjs_idxs[subj]["frames"]
        labels = frame_subjs_idxs[subj]["labels"]

        cap = cv.VideoCapture(video_file)

        for i in range(len(frames)):
            for adj_frames in [-20, 0, 20]:
                frame_idx = frames[i] + adj_frames
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print('PASS')
                else:
                    save_dir = neutral_frames_dir
                    if labels[i] == -1:
                        save_dir = disgust_frames_dir
                    elif labels[i] == 1:
                        save_dir = smile_frames_dir
                    cv.imwrite(f"{save_dir}/{subj}_{frames[i] + adj_frames}_2.png", frame)

                frame_idx = frames[i] + adj_frames - (60 * 2)
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print('PASS')
                else:
                    save_dir = neutral_frames_dir
                    if labels[i] == -1:
                        save_dir = disgust_frames_dir
                    elif labels[i] == 1:
                        save_dir = smile_frames_dir
                    cv.imwrite(f"{save_dir}/{subj}_{frames[i] + adj_frames}_1.png", frame)

