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
        
        # 新角点排序逻辑：基于x+y和x-y来确定TL, TR, BR, BL
        sums = corners[:, 0] + corners[:, 1]
        diffs = corners[:, 0] - corners[:, 1]
        tl_idx = np.argmin(sums)
        br_idx = np.argmax(sums)
        tr_idx = np.argmax(diffs)
        bl_idx = np.argmin(diffs)
        
        # 检查索引是否唯一（防止退化情况）
        if len(set([tl_idx, tr_idx, br_idx, bl_idx])) != 4:
            return False, None
        
        sorted_corners = np.array([corners[tl_idx], corners[tr_idx], corners[br_idx], corners[bl_idx]])
        
        # 可选：验证是否凸四边形（防乱序）
        if not cv2.isContourConvex(sorted_corners.astype(np.int32)):
            return False, None
        
        return True, sorted_corners
    
    def post_crop(self, frame, corners, inset_pixels=5):
        """
        先基于角点进行透视矫正，使A4纸成为正矩形，然后在内侧收缩裁切。
        
        Args:
            frame: 输入图像
            corners: A4纸的四个角点 [左上, 右上, 右下, 左下]
            inset_pixels: 向内裁切的像素数
            
        Returns:
            cropped_frame: 裁切后的图像（已矫正）
            new_corners: 调整后的角点坐标（相对于新图像，矩形角点）
        """
        if corners is None or len(corners) != 4:
            return frame, corners
        
        print("Original:", frame.shape)
        
        # A4纸内轮廓比例：(297-40):(210-40) = 257:170
        A4_RATIO = 257 / 170  # 宽:高的比例，约1.51:1
        
        # 计算四边的长度来确定方向
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        
        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_left + height_right) / 2
        
        # 判断当前检测到的四边形是横放还是竖放的A4纸
        is_landscape = avg_width > avg_height
        
        # 根据A4纸的方向和比例计算目标宽高
        if is_landscape:
            # 横向放置的A4纸
            target_height = max(int(avg_width / A4_RATIO), 1)
            target_width = max(int(avg_width), 1)
        else:
            # 纵向放置的A4纸
            target_width = max(int(avg_height / A4_RATIO), 1)
            target_height = max(int(avg_height), 1)
        
        print(f"Detected orientation: {'Landscape' if is_landscape else 'Portrait'}")
        print(f"Using A4 ratio: {A4_RATIO:.2f}, Target size: {target_width}x{target_height}")
        
        # 目标角点：正矩形
        target_corners = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        # 获取透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        # 应用透视变换矫正图像
        corrected_frame = cv2.warpPerspective(frame, perspective_matrix, (target_width, target_height))
        
        # 第二步：在矫正后的图像上向内收缩 inset_pixels
        crop_h, crop_w = corrected_frame.shape[:2]
        
        # 计算收缩后的边界，确保不超出图像范围
        inset_min_x = max(0, inset_pixels)
        inset_max_x = min(crop_w, crop_w - inset_pixels)
        inset_min_y = max(0, inset_pixels)
        inset_max_y = min(crop_h, crop_h - inset_pixels)
        
        # 最终裁切
        cropped_frame = corrected_frame[inset_min_y:inset_max_y, inset_min_x:inset_max_x]
        
        # 调整角点坐标到新图像坐标系（简单矩形角点）
        new_corners = np.array([
            [0, 0],
            [inset_max_x - inset_min_x - 1, 0],
            [inset_max_x - inset_min_x - 1, inset_max_y - inset_min_y - 1],
            [0, inset_max_y - inset_min_y - 1]
        ], dtype=np.float32)
        
        print("Cropped:", cropped_frame.shape)
        print(f"Final aspect ratio: {(inset_max_x - inset_min_x) / (inset_max_y - inset_min_y):.2f}")
        return cropped_frame, new_corners