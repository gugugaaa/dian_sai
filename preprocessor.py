# preprocessor.py
import cv2
import numpy as np

class Preprocessor:
    def __init__(self, K, dist):
        self.K = K
        self.dist = dist

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

    def pre_crop(self, frame):
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
        # 裁剪图像，添加一些边距以确保完整
        margin = 20
        cropped = frame[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
        
        return cropped, True

    def preprocess(self, frame):
        # 去畸变
        undistorted = cv2.undistort(frame, self.K, self.dist)
        # 转换为灰度图
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        # 应用高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测使用Canny
        edges = cv2.Canny(blurred, 50, 150)
        return edges