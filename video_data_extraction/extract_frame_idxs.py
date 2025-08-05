import os
import glob

import cv2
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

steps_frames = 20


if __name__ == "__main__":
    video_files_list = glob.glob("/home/neuron/mnt/a/A/arhiv/emoscroll/**/*.avi")

    for video_path in video_files_list:
        print(video_path)
        subj_dir = os.path.join("../face_emo_datasets/frame_idx", video_path.split('/')[-2])
        os.mkdir(subj_dir)
        out_path = os.path.join(subj_dir, "frame_idx.csv")
        print(out_path)
        cap = cv2.VideoCapture(video_path)

        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx_dict = {
            "time(s.)": [],
            "frame_idx": []
        }
        print(frames_count)

        for frame_idx in range(0, frames_count, steps_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            if not ret:
                print('PASS')
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_idx_dict["time(s.)"].append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                frame_idx_dict["frame_idx"].append(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame_idx_df = pd.DataFrame.from_dict(frame_idx_dict)
        print(frame_idx_df.head(5))

        frame_idx_df.to_csv(out_path, index=False)
