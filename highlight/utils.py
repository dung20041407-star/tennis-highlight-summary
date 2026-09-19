from dataclasses import dataclass, asdict
import json
import math


@dataclass
class Event:
    frame: int
    event_type: str
    player: str | None = None
    confidence: float = 1.0
    extra: dict | None = None


class EventDetector:
    def __init__(self, fps: int):
        self.fps = fps

    # ------------------------------
    # Utility
    # ------------------------------
    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # ------------------------------
    # Bounce Detection
    # ------------------------------
    def detect_bounces(self, ball_track):

        events = []

        # cần tối thiểu 5 frame để kiểm tra đổi hướng
        for frame in range(2, len(ball_track)-2):

            prev = ball_track[frame-1]
            cur = ball_track[frame]
            nxt = ball_track[frame+1]

            if None in prev or None in cur or None in nxt:
                continue

            # vận tốc theo trục Y
            dy_before = cur[1] - prev[1]
            dy_after = nxt[1] - cur[1]

            # vận tốc theo trục X
            dx_before = cur[0] - prev[0]
            dx_after = nxt[0] - cur[0]

            # bóng đi xuống rồi bật lên
            direction_change = dy_before > 0 and dy_after < 0

            # loại bỏ rung nhẹ
            vertical_speed = abs(dy_before) + abs(dy_after)

            if not direction_change:
                continue

            if vertical_speed < 8:
                continue

            confidence = min(vertical_speed / 40, 1.0)

            events.append(
                Event(
                    frame=frame,
                    event_type="Bounce",
                    confidence=round(confidence, 2)
                )
            )

        return events

    # ------------------------------
    # Save JSON
    # ------------------------------
    @staticmethod
    def save_events(events, output_path):

        data = []

        for event in events:
            data.append(asdict(event))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)