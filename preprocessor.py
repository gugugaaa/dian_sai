# preprocessor.py
import cv2
import numpy as np
from collections import deque

class Preprocessor:
    def __init__(self, K, dist, update_interval=20, history_size=50):
        """
        初始化预处理器
        
        Args:
            K: 相机内参矩阵
            dist: 畸变系数
            update_interval: 裁切区域更新间隔（帧数），默认5帧更新一次
            history_size: 历史区域缓存大小，用于计算平均值，默认10个
        """
        self.K = K
        self.dist = dist
        self.update_interval = update_interval
        self.history_size = history_size
        
        # 防抖动相关参数
        self.frame_count = 0
        self.current_crop_region = None  # 当前使用的裁切区域 (x, y, w, h)
        self.region_history = deque(maxlen=history_size)  # 历史检测区域队列
        self.is_region_valid = False    # 当前区域是否有效

    def _calculate_rectangularity(self, contour):
        """
        计算轮廓的矩形度（轮廓面积与其最小外接矩形面积的比值）
        
        Args:
            contour: 输入轮廓
            
        Returns:
            float: 矩形度值，越接近1表示越像矩形
        """
        contour_area = cv2.contourArea(contour)
        if contour_area == 0:
            return 0
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        
        if rect_area == 0:
            return 0
            
        return contour_area / rect_area

    def _find_valid_contour(self, contours, hierarchy):
        """
        从轮廓中找到有效的轮廓（综合考虑面积和矩形度）
        
        Args:
            contours: 检测到的轮廓列表
            hierarchy: 轮廓的层次结构
            
        Returns:
            tuple: (最佳轮廓, 是否找到有效轮廓)
        """
        if not contours or hierarchy is None:
            return None, False
        
        best_contour = None
        best_score = 0
        min_area_threshold = 1000  # 最小面积阈值
        min_rectangularity = 0.7   # 最小矩形度阈值
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 过滤太小的轮廓
            if area < min_area_threshold:
                continue
                
            # 计算矩形度
            rectangularity = self._calculate_rectangularity(contour)
            
            # 过滤矩形度太低的轮廓
            if rectangularity < min_rectangularity:
                continue
            
            # 检查是否有子轮廓（内轮廓）- A4纸通常有黑边
            has_inner_contour = hierarchy[0][i][2] != -1

            # 综合评分：面积权重0.6，矩形度权重0.3，有内轮廓加分0.1
            area_score = min(area / 50000, 1.0)  # 归一化面积分数
            inner_bonus = 0.1 if has_inner_contour else 0
            total_score = 0.6 * area_score + 0.3 * rectangularity + inner_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_contour = contour
        
        if best_contour is None:
            return None, False
            
        return best_contour, True

    def _should_update_region(self):
        """
        判断是否应该更新裁切区域
        
        Returns:
            bool: 是否需要更新区域
        """
        self.frame_count += 1
        return (self.frame_count % self.update_interval == 0) or (not self.is_region_valid)

    def _detect_crop_region(self, frame):
        """
        检测新的裁切区域
        
        Args:
            frame: 输入帧
            
        Returns:
            tuple: (裁切区域(x,y,w,h), 是否检测成功)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 应用高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        # 查找轮廓，使用RETR_TREE获取层次结构
        # ！！重要：找到的通常是A4纸黑边的内轮廓
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选有效轮廓
        largest_contour, is_valid = self._find_valid_contour(contours, hierarchy)
        
        if not is_valid:
            return None, False
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h), True

    def _is_region_reasonable(self, new_region):
        """
        判断新检测的区域是否合理（与历史平均值差距不会太大）
        
        Args:
            new_region: 新检测的区域 (x, y, w, h)
            
        Returns:
            bool: 区域是否合理
        """
        if not self.region_history:
            return True  # 如果没有历史数据，接受任何合理区域
        
        # 计算当前平均区域
        avg_region = self._calculate_average_region()
        if avg_region is None:
            return True
        
        # 计算新区域与平均区域的差异
        x_diff = abs(new_region[0] - avg_region[0])
        y_diff = abs(new_region[1] - avg_region[1])
        w_diff = abs(new_region[2] - avg_region[2])
        h_diff = abs(new_region[3] - avg_region[3])
        
        # 设置合理的阈值（可以根据实际情况调整）
        max_position_diff = 50  # 位置差异阈值
        max_size_diff = 100     # 尺寸差异阈值
        
        return (x_diff <= max_position_diff and 
                y_diff <= max_position_diff and 
                w_diff <= max_size_diff and 
                h_diff <= max_size_diff)

    def _calculate_average_region(self):
        """
        计算历史区域的平均值
        
        Returns:
            tuple: 平均区域 (x, y, w, h) 或 None
        """
        if not self.region_history:
            return None
        
        # 计算各维度的平均值
        regions_array = np.array(list(self.region_history))
        avg_region = np.mean(regions_array, axis=0).astype(int)
        
        return tuple(avg_region)

    def _update_region_history(self, new_region):
        """
        更新区域历史记录
        
        Args:
            new_region: 新的区域 (x, y, w, h)
        """
        # 只有当新区域合理时才添加到历史记录
        if self._is_region_reasonable(new_region):
            self.region_history.append(new_region)
            return True
        return False

    def pre_crop(self, frame):
        """
        预裁剪图像，带防抖动功能（使用平均值更新）
        
        Args:
            frame: 输入帧
            
        Returns:
            tuple: (裁剪后的图像, 是否成功)
        """
        # 检查是否需要更新裁切区域
        if self._should_update_region():
            new_region, detection_success = self._detect_crop_region(frame)
            
            if detection_success:
                # 尝试更新历史记录
                if self._update_region_history(new_region):
                    # 计算新的平均区域
                    avg_region = self._calculate_average_region()
                    if avg_region is not None:
                        self.current_crop_region = avg_region
                        self.is_region_valid = True
                elif not self.is_region_valid:
                    # 如果没有有效区域且新区域不合理，直接使用新区域
                    self.region_history.append(new_region)
                    self.current_crop_region = new_region
                    self.is_region_valid = True
            elif not self.is_region_valid:
                # 如果检测失败且没有有效的区域，返回失败
                return None, False
        
        # 使用当前区域进行裁切
        if not self.is_region_valid or self.current_crop_region is None:
            return None, False
        
        x, y, w, h = self.current_crop_region
        
        # 裁剪图像，添加一些边距以确保完整
        margin = 20
        frame_height, frame_width = frame.shape[:2]
        
        # 确保裁切区域在图像范围内
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(frame_width, x + w + margin)
        y_end = min(frame_height, y + h + margin)
        
        cropped = frame[y_start:y_end, x_start:x_end]
        
        return cropped, True

    def preprocess(self, frame):
        """
        预处理图像
        
        Args:
            frame: 输入帧
            
        Returns:
            处理后的边缘图像
        """
        # 去畸变
        undistorted = cv2.undistort(frame, self.K, self.dist)
        # 转换为灰度图
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        # 应用高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测使用Canny
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def reset_region_cache(self):
        """
        重置区域缓存，强制在下一帧重新检测
        """
        self.current_crop_region = None
        self.is_region_valid = False
        self.frame_count = 0
        self.region_history.clear()

    def get_region_stats(self):
        """
        获取区域统计信息（用于调试）
        
        Returns:
            dict: 包含当前区域、历史数量等信息
        """
        return {
            'current_region': self.current_crop_region,
            'history_count': len(self.region_history),
            'is_valid': self.is_region_valid,
            'frame_count': self.frame_count
        }