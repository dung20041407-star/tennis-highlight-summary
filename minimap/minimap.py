import cv2
import numpy as np


class MiniMap:
    def __init__(self):
        # Kích thước minimap giống tennis-trace
        self.width = 220
        self.height = 360

        self.margin = 15

        # Tỷ lệ sân tennis chuẩn (giống repo)
        ratio = 1097 / 2377

        self.court_height = int(self.height * 0.88)
        self.court_width = int(self.court_height * ratio)

        self.x_offset = (self.width - self.court_width) // 2
        self.y_offset = (self.height - self.court_height) // 2

        # 4 góc sân trên minimap
        self.courtTL = (self.x_offset, self.y_offset)
        self.courtTR = (self.x_offset + self.court_width, self.y_offset)
        self.courtBL = (self.x_offset, self.y_offset + self.court_height)
        self.courtBR = (
            self.x_offset + self.court_width,
            self.y_offset + self.court_height,
        )

    # Tính Homography từ 4 góc sân
    def get_homography(self, court_points):

        src = np.float32([
            court_points[0],
            court_points[1],
            court_points[2],
            court_points[3],
        ])

        dst = np.float32([
            self.courtTL,
            self.courtTR,
            self.courtBL,
            self.courtBR,
        ])

        H = cv2.getPerspectiveTransform(src, dst)

        return H

    # Chiếu một điểm từ video -> minimap
    def project_point(self, H, point):

        if point is None:
            return None

        if point[0] is None or point[1] is None:
            return None

        pts = np.float32([[point]])
        mapped = cv2.perspectiveTransform(pts, H)[0][0]

        return int(mapped[0]), int(mapped[1])

    # Lấy vị trí chân từ bbox YOLO
    def player_feet(self, bbox):

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox

        foot_x = (x1 + x2) / 2
        foot_y = y2

        return foot_x, foot_y

    # Vẽ sân tennis
    def draw_court(self):

        court = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        court[:] = (35, 120, 35)

        # Khung ngoài
        cv2.rectangle(
            court,
            self.courtTL,
            self.courtBR,
            (255, 255, 255),
            2,
        )

        # Lưới
        net_y = self.y_offset + self.court_height // 2

        cv2.line(
            court,
            (self.x_offset, net_y),
            (self.x_offset + self.court_width, net_y),
            (255, 255, 255),
            2,
        )

        alley = int(self.court_width * 0.124886)

        left_single = self.x_offset + alley
        right_single = self.x_offset + self.court_width - alley

        # Singles lines
        cv2.line(
            court,
            (left_single, self.y_offset),
            (left_single, self.y_offset + self.court_height),
            (255, 255, 255),
            2,
        )

        cv2.line(
            court,
            (right_single, self.y_offset),
            (right_single, self.y_offset + self.court_height),
            (255, 255, 255),
            2,
        )

        service = int(self.court_height * 0.23054)

        top_service = self.y_offset + service
        bottom_service = self.y_offset + self.court_height - service

        # Service lines
        cv2.line(
            court,
            (left_single, top_service),
            (right_single, top_service),
            (255, 255, 255),
            2,
        )

        cv2.line(
            court,
            (left_single, bottom_service),
            (right_single, bottom_service),
            (255, 255, 255),
            2,
        )

        center_x = self.x_offset + self.court_width // 2

        # Center line
        cv2.line(
            court,
            (center_x, top_service),
            (center_x, bottom_service),
            (255, 255, 255),
            2,
        )

        return court

    # Vẽ minimap lên frame
    def draw(self, frame, H, players, ball):

        minimap = self.draw_court()

        # Player
        colors = {
            "Player 1": (255, 60, 60),
            "Player 2": (60, 180, 255),
        }

        for player_id, bbox in players.items():

            feet = self.player_feet(bbox)

            point = self.project_point(H, feet)

            if point is None:
                continue

            cv2.circle(
                minimap,
                point,
                6,
                colors.get(player_id, (255, 255, 0)),
                -1,
            )

        # Ball
        point = self.project_point(H, ball)

        if point is not None:
            cv2.circle(
                minimap,
                point,
                4,
                (0, 255, 255),
                -1,
            )

        # Ghép minimap vào góc phải trên
        x = frame.shape[1] - self.width - self.margin
        y = self.margin

        roi = frame[y:y + self.height, x:x + self.width]

        blended = cv2.addWeighted(roi, 0.35, minimap, 0.65, 0)

        frame[y:y + self.height, x:x + self.width] = blended

        return frame