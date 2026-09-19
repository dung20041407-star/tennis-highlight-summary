import os
import sys
import cv2
import importlib.util

from highlight.event_detector import EventDetector
from highlight.rally_detector import RallyDetector
from highlight.video_cutter import VideoCutter

BASE_DIR = os.path.dirname(__file__)

tracknet_path = os.path.join(BASE_DIR, "TrackNet")
sys.path.insert(0, tracknet_path)

spec = importlib.util.spec_from_file_location(
    "tracknet_infer",
    os.path.join(tracknet_path, "infer_on_video.py")
)
tracknet_infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracknet_infer)

infer_ball_model = tracknet_infer.infer_ball_model
read_video = tracknet_infer.read_video

sys.path.pop(0)

for name in ["model", "postprocess", "general"]:
    sys.modules.pop(name, None)

player_path = os.path.join(BASE_DIR, "PlayerDetector")
sys.path.insert(0, player_path)

spec = importlib.util.spec_from_file_location(
    "player_infer",
    os.path.join(player_path, "infer_on_video.py")
)
player_infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(player_infer)

infer_player_model = player_infer.infer_player_model

sys.path.pop(0)

court_path = os.path.join(BASE_DIR, "TennisCourtDetector")
sys.path.insert(0, court_path)

spec = importlib.util.spec_from_file_location(
    "court_infer",
    os.path.join(court_path, "infer_in_video.py")
)
court_infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(court_infer)

infer_court_model = court_infer.infer_court_model

sys.path.pop(0)

for name in ["tracknet", "postprocess", "homography", "utils"]:
    sys.modules.pop(name, None)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from minimap import MiniMap

INPUT_DIR = os.path.join(BASE_DIR, "input_video")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_video")

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]

if len(video_files) == 0:
    raise FileNotFoundError("Không tìm thấy video trong input_video.")

VIDEO_PATH = os.path.join(INPUT_DIR, video_files[0])

BALL_MODEL = os.path.join(BASE_DIR, "TrackNet", "model_best.pt")
PLAYER_MODEL = os.path.join(BASE_DIR, "PlayerDetector", "yolov8n.pt")
COURT_MODEL = os.path.join(
    BASE_DIR,
    "TennisCourtDetector",
    "model_tennis_court_det.pt"
)

video_name = os.path.splitext(video_files[0])[0]

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{video_name}_result.mp4"
)

COURT_LINES = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
]

BALL_HISTORY = 6

def main():

    print("Reading video...")
    frames, fps = read_video(VIDEO_PATH)

    print("Running TrackNet...")
    ball_track = infer_ball_model(frames, BALL_MODEL)

    print("Running TennisCourtDetector...")
    court_track = infer_court_model(
        frames,
        COURT_MODEL,
        use_refine_kps=False,
        use_homography=False,
    )

    print("Running Player Detector...")
    player_track = infer_player_model(
        frames,
        PLAYER_MODEL,
        court_track
    )

    print("Detecting bounce events...")

    event_detector = EventDetector(fps)

    bounce_events = event_detector.detect_bounces(
        ball_track,
        court_track
    )

    event_detector.save_events(
        bounce_events,
        os.path.join(OUTPUT_DIR, "events.json")
    )

    print(f"Detected {len(bounce_events)} bounce events.")

    print("Detecting rallies...")

    rally_detector = RallyDetector(fps)

    rallies = rally_detector.detect_rallies(
        ball_track,
        court_track
    )

    print(f"Detected {len(rallies)} rallies.")

    for i, rally in enumerate(rallies, start=1):
        print(
            f"Rally {i}: start={rally.start_frame}, "
            f"end={rally.end_frame}, "
            f"duration={rally.duration_frames}"
        )

    minimap = MiniMap()

    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (w, h),
    )

    print("Drawing output...")

    for frame_idx, (frame, court_points, players) in enumerate(
        zip(frames, court_track, player_track)
    ):

        image = frame.copy()

        for a, b in COURT_LINES:

            if (
                court_points[a] is None
                or court_points[b] is None
                or court_points[a][0] is None
                or court_points[a][1] is None
                or court_points[b][0] is None
                or court_points[b][1] is None
            ):
                continue

            cv2.line(
                image,
                tuple(map(int, court_points[a])),
                tuple(map(int, court_points[b])),
                (0, 255, 0),
                2,
            )

        for idx, point in enumerate(court_points):

            if point is None or point[0] is None or point[1] is None:
                continue

            x, y = map(int, point)

            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            cv2.putText(
                image,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )

        for label, bbox in players.items():

            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        for i in range(BALL_HISTORY - 1, -1, -1):

            idx = frame_idx - i

            if idx < 0:
                continue

            ball = ball_track[idx]

            if ball is None:
                continue

            x, y = ball

            if x is None or y is None:
                continue

            radius = max(2, 8 - i)
            intensity = int(255 * (1 - i / BALL_HISTORY))

            cv2.circle(
                image,
                (int(x), int(y)),
                radius,
                (0, 0, intensity),
                -1,
            )

        if all(
            court_points[i] is not None
            and court_points[i][0] is not None
            and court_points[i][1] is not None
            for i in range(4)
        ):

            H = minimap.get_homography(court_points)

            ball = ball_track[frame_idx]

            image = minimap.draw(
                frame=image,
                H=H,
                players=players,
                ball=ball,
            )

        out.write(image)

    out.release()

    print("Saved edited video:", OUTPUT_PATH)

    print("Cutting rally videos...")

    video_cutter = VideoCutter(fps)

    rally_output_dir = os.path.join(
        OUTPUT_DIR,
        "rallies"
    )

    video_cutter.cut_rallies(
        OUTPUT_PATH,
        rallies,
        rally_output_dir,
    )

    print("Done!")
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()