from dataclasses import dataclass


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    duration_frames: int


class RallyDetector:

    def __init__(self, fps):
        self.fps = fps

        # nếu mất bóng quá 1 giây thì coi là kết thúc rally
        self.max_missing_frames = int(fps)

        # rally tối thiểu 2 giây
        self.min_rally_frames = int(fps * 2)

    def detect_rallies(self, ball_track, court_track):

        rallies = []

        in_rally = False
        start = 0
        missing = 0

        for frame in range(len(ball_track)):

            # Đếm số keypoint sân nhìn thấy
            visible_points = sum(
                p is not None and p[0] is not None and p[1] is not None
                for p in court_track[frame]
            )

            # Thấy đủ 14 điểm sân
            court_visible = visible_points == 14

            if court_visible:
                if not in_rally:
                    start = frame
                    in_rally = True

                missing = 0

            else:
                if in_rally:
                    missing += 1

                    if missing > self.max_missing_frames:
                        end = frame - missing

                        if end - start >= self.min_rally_frames:
                            rallies.append(
                                Rally(
                                    start_frame=start,
                                    end_frame=end,
                                    duration_frames=end - start,
                                )
                            )

                        in_rally = False
                        missing = 0

        if in_rally:
            end = len(ball_track) - 1

            if end - start >= self.min_rally_frames:
                rallies.append(
                    Rally(
                        start_frame=start,
                        end_frame=end,
                        duration_frames=end - start,
                    )
                )

        return rallies