import cv2
import argparse
import math
from ultralytics import YOLO

PERSON_CLASS = 0


# ======================================================
# Read video
# ======================================================
def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

    cap.release()

    return frames, fps


# ======================================================
# Save video (test riêng PlayerDetector)
# ======================================================
def write_track(frames, player_track, output_path, fps):

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for frame, players in zip(frames, player_track):

        image = frame.copy()

        for label, bbox in players.items():

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        writer.write(image)

    writer.release()


# ======================================================
# Utils
# ======================================================
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ======================================================
# Player detector dùng trong pipeline
# ======================================================
def infer_player_model(frames, model_path, court_track):
    """
    court_track = output của TennisCourtDetector

    Return:
    [
        {
            "Player 1": bbox,
            "Player 2": bbox
        },
        ...
    ]
    """

    model = YOLO(model_path)

    frame_tracks = []

    # --------------------------------------------------
    # Pass 1: Detect + Track tất cả person
    # --------------------------------------------------
    for frame in frames:

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            verbose=False,
        )

        result = results[0]
        tracks = []

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                tracks.append(
                    {
                        "id": track_id,
                        "bbox": (x1, y1, x2, y2),
                        "center": bbox_center((x1, y1, x2, y2)),
                    }
                )

        frame_tracks.append(tracks)

    # --------------------------------------------------
    # Pass 2: Chọn 2 người gần sân nhất ở frame đầu
    # --------------------------------------------------
    first_tracks = frame_tracks[0]
    first_keypoints = court_track[0]

    ranking = []

    for track in first_tracks:

        player_center = track["center"]

        min_dist = float("inf")

        for kp in first_keypoints:

            if kp[0] is None:
                continue

            d = distance(player_center, kp)

            if d < min_dist:
                min_dist = d

        ranking.append((track["id"], min_dist))

    ranking.sort(key=lambda x: x[1])

    if len(ranking) < 2:
        return [{"Player 1": None, "Player 2": None} for _ in frames]

    selected_ids = {
        ranking[0][0],
        ranking[1][0],
    }

    # --------------------------------------------------
    # Pass 3: Giữ 2 ID này trong toàn video
    # --------------------------------------------------
    player_track = []

    half_height = frames[0].shape[0] / 2

    for tracks in frame_tracks:

        selected = []

        for track in tracks:
            if track["id"] in selected_ids:
                selected.append(track)

        player1 = None
        player2 = None

        if len(selected) == 1:

            cy = selected[0]["center"][1]

            if cy < half_height:
                player1 = selected[0]["bbox"]
            else:
                player2 = selected[0]["bbox"]

        elif len(selected) >= 2:

            selected.sort(key=lambda t: t["center"][1])

            player1 = selected[0]["bbox"]
            player2 = selected[1]["bbox"]

        player_track.append(
            {
                "Player 1": player1,
                "Player 2": player2,
            }
        )

    return player_track


# ======================================================
# Test riêng module PlayerDetector
# ======================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="yolov8n.pt")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    frames, fps = read_video(args.video_path)

    print("Module này dùng trong pipeline.py vì cần court_track từ TennisCourtDetector.")