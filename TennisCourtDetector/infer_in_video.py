import os
import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse


def read_video(path_video):
    """ Read video file
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def write_video(imgs_new, fps, path_output_video):
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(
        path_output_video,
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps,
        (width, height)
    )

    for frame in imgs_new:
        out.write(frame)

    out.release()


def infer_court_model(
    frames,
    model_path,
    use_refine_kps=False,
    use_homography=False
):

    model = BallTrackerNet(out_channels=15)

    device = "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360

    court_track = []

    for image in tqdm(frames):

        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        inp = img.astype(np.float32) / 255.0
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []

        for kps_num in range(14):

            heatmap = (pred[kps_num] * 255).astype(np.uint8)

            x_pred, y_pred = postprocess(
                heatmap,
                low_thresh=170,
                max_radius=25
            )

            if (
                use_refine_kps
                and kps_num not in [8, 12, 9]
                and x_pred is not None
                and y_pred is not None
            ):
                x_pred, y_pred = refine_kps(
                    image,
                    int(y_pred),
                    int(x_pred)
                )

            points.append((x_pred, y_pred))

        if use_homography:
            matrix_trans = get_trans_matrix(points)

            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        court_track.append(points)

    return court_track


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to model")
    parser.add_argument("--input_path", type=str, help="path to input video")
    parser.add_argument("--output_path", type=str, help="path to output video")
    parser.add_argument("--use_refine_kps", action="store_true")
    parser.add_argument("--use_homography", action="store_true")

    args = parser.parse_args()

    frames, fps = read_video(args.input_path)

    court_track = infer_court_model(
        frames,
        args.model_path,
        args.use_refine_kps,
        args.use_homography
    )

    frames_upd = []

    for frame, points in zip(frames, court_track):

        image = frame.copy()

        for point in points:
            if point[0] is not None:
                image = cv2.circle(
                    image,
                    (int(point[0]), int(point[1])),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=10
                )

        frames_upd.append(image)

    write_video(frames_upd, fps, args.output_path)