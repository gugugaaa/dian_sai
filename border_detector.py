# border_detector.py
import cv2
import numpy as np

class BorderDetector:
    def __init__(self, update_interval=20):
        """
        初始化边框检测器
        
        Args:
            update_interval: 角点更新间隔帧数，每N帧重新检测一次角点
        """
        self.update_interval = update_interval
        self.frame_counter = 0
        self.stable_corners = None  # 当前稳定的角点
        self.need_detection = True  # 是否需要重新检测
        
    def detect_border(self, edges, frame):
        """
        检测边框，每update_interval帧才重新检测一次
        """
        self.frame_counter += 1
        
        # 如果还没到更新时间且有稳定角点，直接返回
        if not self.need_detection and self.stable_corners is not None:
            if self.frame_counter % self.update_interval != 0:
                return True, self.stable_corners
        
        # 需要重新检测角点
        success, new_corners = self._detect_border_internal(edges, frame)
        
        if success and new_corners is not None:
            # 更新稳定角点
            self.stable_corners = new_corners.copy()
            self.need_detection = False
            print(f"Frame {self.frame_counter}: Updated corners")
            return True, self.stable_corners
        else:
            # 检测失败，继续使用之前的稳定角点
            if self.stable_corners is not None:
                print(f"Frame {self.frame_counter}: Detection failed, using stable corners")
                return True, self.stable_corners
            else:
                print(f"Frame {self.frame_counter}: No stable corners available")
                return False, None
    
    def _detect_border_internal(self, edges, frame):
        """
        内部边框检测逻辑（原来的detect_border逻辑）
        """
        # 查找轮廓
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None
        
        # 找到最大的轮廓和其索引
        areas = [cv2.contourArea(c) for c in contours]
        largest_idx = np.argmax(areas)
        largest_contour = contours[largest_idx]
        
        # 逼近轮廓为多边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:
            return False, None  # 不是四边形
        
        # 检查最大轮廓内部是否有子轮廓
        has_inner_contours = False
        if hierarchy is not None:
            # 查找当前轮廓的第一个子轮廓
            first_child = hierarchy[0][largest_idx][2]  # [2]是第一个子轮廓索引
            if first_child != -1:
                has_inner_contours = True
        
        if not has_inner_contours:
            return False, None  # 内部没有轮廓，可能是A4纸上的目标
        
        # 检查是否为A4纸外框
        current_contour = largest_contour
        h, w = frame.shape[:2]
        
        # 计算当前轮廓的边界框
        rect = cv2.boundingRect(current_contour)
        rect_width, rect_height = rect[2], rect[3]
        rect_ratio = rect_width / rect_height if rect_height > 0 else 0
        
        # A4纸外框比例约为 297:210 ≈ 1.41
        A4_OUTER_RATIO = 297 / 210
        # A4纸内框比例约为 257:170 ≈ 1.51
        A4_INNER_RATIO = 257 / 170
        
        # 计算轮廓面积占图像面积的比例
        contour_area = cv2.contourArea(current_contour)
        frame_area = w * h
        area_ratio = contour_area / frame_area
        
        # 判断是否为A4纸外框：宽高比在1.3-1.6之间，且面积占比较大
        is_outer_frame = (1.3 <= rect_ratio <= 1.6 and area_ratio > 0.3)
        
        if is_outer_frame and hierarchy is not None:
            # 尝试找到内部的第一个四边形轮廓作为真正的边框
            first_child = hierarchy[0][largest_idx][2]
            if first_child != -1:
                inner_contour = contours[first_child]
                
                # 检查内部轮廓是否为四边形
                inner_epsilon = 0.02 * cv2.arcLength(inner_contour, True)
                inner_approx = cv2.approxPolyDP(inner_contour, inner_epsilon, True)
                
                if len(inner_approx) == 4:
                    # 验证内部轮廓的比例是否符合A4纸内框
                    inner_rect = cv2.boundingRect(inner_contour)
                    inner_ratio = inner_rect[2] / inner_rect[3] if inner_rect[3] > 0 else 0
                    
                    # 检查内部轮廓的宽高比是否在1.3-1.6范围内
                    if 1.3 <= inner_ratio <= 1.6:  # 稍微放宽范围
                        # 使用内部轮廓作为真正的边框
                        current_contour = inner_contour
                        approx = inner_approx
        
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
    
    def force_update(self):
        """强制在下一帧重新检测角点"""
        self.need_detection = True
        
    def reset(self):
        """重置检测器状态"""
        self.frame_counter = 0
        self.stable_corners = None
        self.need_detection = True
        
    def set_update_interval(self, interval):
        """动态调整更新间隔"""
        self.update_interval = interval
        
    def get_current_corners(self):
        """获取当前稳定的角点"""
        return self.stable_corners
    
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