from model import BallTrackerNet
import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance


# ======================================================
# Read video
# ======================================================
def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


# ======================================================
# Run TrackNet
# ======================================================
def infer_model(frames, model, device):

    MODEL_WIDTH = 640
    MODEL_HEIGHT = 360

    dists = [-1] * 2
    ball_track = [(None, None)] * 2

    for num in tqdm(range(2, len(frames))):

        # Kích thước frame gốc
        orig_h, orig_w = frames[num].shape[:2]

        # Resize về kích thước TrackNet
        img = cv2.resize(frames[num], (MODEL_WIDTH, MODEL_HEIGHT))
        img_prev = cv2.resize(frames[num - 1], (MODEL_WIDTH, MODEL_HEIGHT))
        img_preprev = cv2.resize(frames[num - 2], (MODEL_WIDTH, MODEL_HEIGHT))

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().to(device))

        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)

        # postprocess() đã nhân scale=2 -> hệ tọa độ 1280x720
        if x_pred is None or y_pred is None:
            ball_track.append((None, None))
        else:
            x = float(x_pred) * orig_w / 1280.0
            y = float(y_pred) * orig_h / 720.0
            ball_track.append((x, y))

        if (
            ball_track[-1][0] is not None
            and ball_track[-2][0] is not None
        ):
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1

        dists.append(dist)

    return ball_track, dists


# ======================================================
# Remove outliers
# ======================================================
def remove_outliers(ball_track, dists, max_dist=100):

    outliers = list(np.where(np.array(dists) > max_dist)[0])

    for i in outliers:

        if (dists[i + 1] > max_dist) or (dists[i + 1] == -1):
            ball_track[i] = (None, None)
            if i in outliers:
                outliers.remove(i)

        elif dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)

    return ball_track


# ======================================================
# Split tracks
# ======================================================
def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):

    list_det = [0 if x[0] is not None else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []

    for i, (k, l) in enumerate(groups):

        if (k == 1) and (i > 0) and (i < len(groups) - 1):

            dist = distance.euclidean(
                ball_track[cursor - 1],
                ball_track[cursor + l]
            )

            if (l >= max_gap) or (dist / l > max_dist_gap):

                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1

        cursor += l

    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])

    return result


# ======================================================
# Interpolation
# ======================================================
def interpolation(coords):

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([p[0] if p[0] is not None else np.nan for p in coords])
    y = np.array([p[1] if p[1] is not None else np.nan for p in coords])

    nans, idx = nan_helper(x)
    x[nans] = np.interp(idx(nans), idx(~nans), x[~nans])

    nans, idx = nan_helper(y)
    y[nans] = np.interp(idx(nans), idx(~nans), y[~nans])

    return list(zip(x, y))


# ======================================================
# Write result video
# ======================================================
def write_track(frames, ball_track, path_output_video, fps, trace=20):

    height, width = frames[0].shape[:2]

    out = cv2.VideoWriter(
        path_output_video,
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )

    for num in range(len(frames)):

        frame = frames[num].copy()

        for i in range(trace):

            idx = num - i

            if idx < 0:
                break

            x, y = ball_track[idx]

            if x is None or y is None:
                break

            cv2.circle(
                frame,
                (int(x), int(y)),
                radius=max(2, 8 - i),
                color=(0, 0, 255),
                thickness=-1,
            )

        out.write(frame)

    out.release()


# ======================================================
# Public function for pipeline
# ======================================================
def infer_ball_model(frames, model_path, use_extrapolation=False):

    model = BallTrackerNet()

    device = "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    ball_track, dists = infer_model(frames, model, device)
    ball_track = remove_outliers(ball_track, dists)

    if use_extrapolation:
        subtracks = split_track(ball_track)

        for start, end in subtracks:
            ball_track[start:end] = interpolation(ball_track[start:end])

    return ball_track


# ======================================================
# Standalone test
# ======================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--video_out_path", type=str)
    parser.add_argument("--extrapolation", action="store_true")

    args = parser.parse_args()

    frames, fps = read_video(args.video_path)

    ball_track = infer_ball_model(
        frames,
        args.model_path,
        args.extrapolation
    )

    write_track(frames, ball_track, args.video_out_path, fps)