import os
import cv2


class VideoCutter:

    def __init__(self, fps):
        self.fps = fps

        # thêm context trước và sau rally
        self.pre_seconds = 2
        self.post_seconds = 2

    def cut_rallies(self, video_path, rallies, output_dir):

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        saved_paths = []

        for i, rally in enumerate(rallies, start=1):

            start = max(
                0,
                rally.start_frame - int(self.pre_seconds * self.fps)
            )

            end = rally.end_frame + int(self.post_seconds * self.fps)

            save_path = os.path.join(
                output_dir,
                f"rally_{i:03d}.mp4"
            )

            writer = cv2.VideoWriter(
                save_path,
                fourcc,
                self.fps,
                (width, height),
            )

            # Nhảy đến frame bắt đầu
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            current = start

            while current <= end:

                ret, frame = cap.read()

                if not ret:
                    break

                writer.write(frame)
                current += 1

            writer.release()

            saved_paths.append(save_path)

            print(
                f"[Saved] Rally {i}: "
                f"{start} -> {current - 1} "
                f"({(current - start) / self.fps:.2f}s)"
            )

        cap.release()

        return saved_paths