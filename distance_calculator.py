# distance_calculator.py
import cv2
import numpy as np

class DistanceCalculator:
    def __init__(self, K):
        self.K = K
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

    def calculate_D(self, image_points):
        # 使用solvePnP计算旋转和平移向量
        success, rvec, tvec = cv2.solvePnP(self.object_points, image_points, self.K, None)
        if not success:
            return None
        
        # 距离D是tvec的Z分量（相机到平面的距离）
        D = tvec[2][0]  # cm
        return D