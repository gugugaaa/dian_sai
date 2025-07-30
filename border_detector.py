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
        基于A4纸角点的外接矩形裁切，然后向内收缩
        
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
        
        print("Original:", frame.shape)
        
        # 计算四个角点的外接矩形
        min_x = int(np.min(corners[:, 0]))
        max_x = int(np.max(corners[:, 0]))
        min_y = int(np.min(corners[:, 1]))
        max_y = int(np.max(corners[:, 1]))
        
        # 确保边界在图像范围内
        h, w = frame.shape[:2]
        min_x = max(0, min_x)
        max_x = min(w, max_x)
        min_y = max(0, min_y)
        max_y = min(h, max_y)
        
        # 第一步：裁切外接矩形
        rect_crop = frame[min_y:max_y, min_x:max_x]
        
        # 第二步：在矩形基础上向内收缩 inset_pixels
        crop_h, crop_w = rect_crop.shape[:2]
        
        # 计算收缩后的边界，确保不超出图像范围
        inset_min_x = max(0, inset_pixels)
        inset_max_x = min(crop_w, crop_w - inset_pixels)
        inset_min_y = max(0, inset_pixels)
        inset_max_y = min(crop_h, crop_h - inset_pixels)
        
        # 最终裁切
        cropped_frame = rect_crop[inset_min_y:inset_max_y, inset_min_x:inset_max_x]
        
        # 调整角点坐标到新图像坐标系
        adjusted_corners = corners.copy()
        # 先减去外接矩形的偏移
        adjusted_corners[:, 0] -= min_x
        adjusted_corners[:, 1] -= min_y
        # 再减去内收缩的偏移
        adjusted_corners[:, 0] -= inset_min_x
        adjusted_corners[:, 1] -= inset_min_y
        
        print("Cropped:", cropped_frame.shape)
        return cropped_frame, adjusted_corners.astype(np.float32)