from dataclasses import dataclass, asdict
import json
import numpy as np
import cv2
from scipy.signal import savgol_filter


@dataclass
class BounceEvent:
    frame: int
    x: float
    y: float
    type: str = "bounce"


class EventDetector:

    def __init__(self, fps):
        self.fps = fps

        # ===== Savitzky-Golay =====
        self.window = 11          # số frame làm mượt
        self.poly_order = 3

        # ===== Bounce filter =====
        self.min_gap_frames = 10  # khoảng cách tối thiểu giữa 2 bounce
        self.velocity_threshold = 4.0

    # =====================================================
    # Kiểm tra bóng nằm trong sân
    # =====================================================
    def is_inside_court(self, point, court_points):

        if point is None:
            return False

        x, y = point

        corners = []

        for idx in [0, 1, 2, 3]:
            kp = court_points[idx]

            if kp is None:
                return False

            if kp[0] is None or kp[1] is None:
                return False

            corners.append(kp)

        polygon = np.array(corners, dtype=np.float32)

        return cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        ) >= 0

    # =====================================================
    # Làm mượt quỹ đạo
    # =====================================================
    def smooth_track(self, ball_track):

        xs = np.array([
            np.nan if p is None or p[0] is None else p[0]
            for p in ball_track
        ], dtype=float)

        ys = np.array([
            np.nan if p is None or p[1] is None else p[1]
            for p in ball_track
        ], dtype=float)

        # Nội suy các đoạn mất bóng
        valid = ~np.isnan(xs)

        if valid.sum() < self.window:
            return xs, ys

        idx = np.arange(len(xs))

        xs = np.interp(idx, idx[valid], xs[valid])
        ys = np.interp(idx, idx[valid], ys[valid])

        xs = savgol_filter(xs, self.window, self.poly_order)
        ys = savgol_filter(ys, self.window, self.poly_order)

        return xs, ys

    # =====================================================
    # Bounce Detection
    # =====================================================
    def detect_bounces(self, ball_track, court_track):

        smooth_x, smooth_y = self.smooth_track(ball_track)

        velocity_y = np.gradient(smooth_y)

        bounce_events = []

        last_bounce = -999

        for i in range(2, len(ball_track) - 2):

            if ball_track[i] is None:
                continue

            if i - last_bounce < self.min_gap_frames:
                continue

            # Đổi hướng từ đi xuống sang đi lên
            descending = velocity_y[i - 1] > self.velocity_threshold
            ascending = velocity_y[i + 1] < -self.velocity_threshold

            if not (descending and ascending):
                continue

            x = smooth_x[i]
            y = smooth_y[i]

            if not self.is_inside_court((x, y), court_track[i]):
                continue

            bounce_events.append(
                BounceEvent(
                    frame=i,
                    x=float(x),
                    y=float(y)
                )
            )

            last_bounce = i

        return bounce_events

    # =====================================================
    # Save JSON
    # =====================================================
    def save_events(self, events, save_path):

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {"events": [asdict(e) for e in events]},
                f,
                indent=4,
                ensure_ascii=False
            )