# distance_calculator.py
import cv2
import numpy as np

class DistanceCalculator:
    def __init__(self):
        # 内边框识别：A4外边框每边减去2cm
        self.real_width = 17.0  # cm (21.0 - 2*2)
        self.real_height = 25.7  # cm (29.7 - 2*2)
        # 3D世界点：左上(-w/2, h/2, 0), 右上(w/2, h/2, 0), 右下(w/2, -h/2, 0), 左下(-w/2, -h/2, 0)
        half_w = self.real_width / 2
        half_h = self.real_height / 2
        self.object_points = np.array([
            [-half_w, half_h, 0],
            [half_w, half_h, 0],
            [half_w, -half_h, 0],
            [-half_w, -half_h, 0]
        ], dtype=np.float32)

    def calculate_D(self, image_points, K):
        success, rvec, tvec = cv2.solvePnP(self.object_points, image_points, K, None)
        if not success:
            return None, None, None
        D = tvec[2][0]  # cm
        theta_deg = np.linalg.norm(rvec) * (180 / np.pi)  # 粗略夹角
        return D, theta_deg