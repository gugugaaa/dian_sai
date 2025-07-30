# border_detector.py
import cv2
import numpy as np

class BorderDetector:
    def detect_border(self, edges, frame):
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None
        
        # 找到最大的轮廓（假设是A4边框的内侧）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 逼近轮廓为多边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:
            return False, None  # 不是四边形
        
        # 计算凸包
        hull = cv2.convexHull(approx)
        
        # 获取角点
        corners = hull.reshape(-1, 2).astype(np.float32)
        if len(corners) != 4:
            return False, None
        
        # 排序角点：假设顺时针或逆时针，从左上开始
        # 先按x+y排序找到左上，然后顺时针
        sum_coords = np.sum(corners, axis=1)
        top_left = corners[np.argmin(sum_coords)]
        bottom_right = corners[np.argmax(sum_coords)]
        diff_coords = np.diff(corners, axis=1)
        top_right = corners[np.argmax(diff_coords)]
        bottom_left = corners[np.argmin(diff_coords)]
        
        sorted_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        
        return True, sorted_corners
    
    def post_crop(self, frame, corners, inset_pixels=5):
        """
        基于A4纸角点向内裁切，避免边框干扰形状检测
        
        Args:
            frame: 输入图像
            corners: A4纸的四个角点 [左上, 右上, 右下, 左下]
            inset_pixels: 向内裁切的像素数
            
        Returns:
            cropped_frame: 裁切后的图像
            new_corners: 调整后的角点坐标（相对于新图像）
        """
        if corners is None or len(corners) != 4:
            return frame, corners
        
        # 向内收缩角点
        top_left, top_right, bottom_right, bottom_left = corners
        
        # 计算向内偏移的角点
        # 左上角向右下偏移
        new_top_left = top_left + [inset_pixels, inset_pixels]
        # 右上角向左下偏移  
        new_top_right = top_right + [-inset_pixels, inset_pixels]
        # 右下角向左上偏移
        new_bottom_right = bottom_right + [-inset_pixels, -inset_pixels]
        # 左下角向右上偏移
        new_bottom_left = bottom_left + [inset_pixels, -inset_pixels]
        
        # 确保新角点在图像范围内
        h, w = frame.shape[:2]
        new_corners = np.array([new_top_left, new_top_right, new_bottom_right, new_bottom_left])
        new_corners[:, 0] = np.clip(new_corners[:, 0], 0, w-1)
        new_corners[:, 1] = np.clip(new_corners[:, 1], 0, h-1)
        
        # 计算裁切区域的边界
        min_x = int(np.min(new_corners[:, 0]))
        max_x = int(np.max(new_corners[:, 0]))
        min_y = int(np.min(new_corners[:, 1]))
        max_y = int(np.max(new_corners[:, 1]))
        
        # 裁切图像
        cropped_frame = frame[min_y:max_y, min_x:max_x]
        
        # 调整角点坐标到新图像坐标系
        adjusted_corners = new_corners.copy()
        adjusted_corners[:, 0] -= min_x
        adjusted_corners[:, 1] -= min_y
        
        return cropped_frame, adjusted_corners.astype(np.float32)