# shape_detector.py
import cv2
import numpy as np

class ShapeDetector:
    def _detect_polygonal(self, approx, edges):
        num_sides = len(approx)
        print("num_sides:", num_sides)
        
        # 霍夫直线检测
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
        num_hough_lines = 0
        if lines is not None:
            thetas = lines[:, 0, 1]
            thetas = np.sort(thetas)
            groups = []
            current = [thetas[0]]
            for t in thetas[1:]:
                if t - current[-1] < 0.1:
                    current.append(t)
                else:
                    groups.append(current)
                    current = [t]
            groups.append(current)
            num_hough_lines = len(groups)

        print("num_hough_lines:", num_hough_lines)

        # 判断是否为三角形或正方形
        # 只要霍夫直线数量或多边形边数其中一个满足条件即可
        if num_sides == 3 or num_hough_lines == 3:
            shape = "triangle"
            shape_params = approx.reshape(-1, 2)
            return shape, shape_params
        elif num_sides == 4 or num_hough_lines == 4:
            shape = "square"
            shape_params = approx.reshape(-1, 2)
            return shape, shape_params
        
        return None, None

    def _detect_circle(self, contour, blurred):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None, None
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity <= 0.75:
            return None, None

        (cx, cy), mr = cv2.minEnclosingCircle(contour)
        cx, cy, mr = int(cx), int(cy), int(mr)

        h_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                     param1=50, param2=25, minRadius=int(mr * 0.5), maxRadius=int(mr * 1.5))
        best_center = (cx, cy)
        best_radius = mr
        if h_circles is not None:
            h_circles = np.round(h_circles[0, :]).astype("int")
            best_score = float('inf')
            for (hx, hy, hr) in h_circles:
                dist = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
                rel_dist = dist / mr if mr > 0 else 0
                rel_r = abs(hr - mr) / mr if mr > 0 else 0
                score = rel_dist + rel_r
                if score < best_score:
                    best_score = score
                    best_center = (hx, hy)
                    best_radius = hr
            if best_score >= 0.3:
                best_center = (cx, cy)
                best_radius = mr

        shape_params = {"center": best_center, "radius": best_radius}
        return "circle", shape_params

    def detect_shape(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, None
        target_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(target_contour, True)
        approx = cv2.approxPolyDP(target_contour, epsilon, True)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        shape, shape_params = self._detect_polygonal(approx, edges)
        num_sides = len(approx) if shape in ["triangle", "square"] else 0
        if shape is None:
            shape, shape_params = self._detect_circle(target_contour, blurred)

        if shape is None:
            return None, 0, None

        if shape == "circle":
            x_pix = 2 * shape_params["radius"]
        else:
            sides = []
            points = shape_params
            for i in range(num_sides):
                p1 = points[i]
                p2 = points[(i + 1) % num_sides]
                side = np.linalg.norm(p1 - p2)
                sides.append(side)
            x_pix = np.mean(sides)

        return shape, x_pix, shape_params

    def draw_shape(self, frame, shape_type, shape_params, color=(0, 255, 0), thickness=2):
        """
        在画面上绘制检测到的形状
        Args:
            frame: 输入图像
            shape_type: 形状类型 ("triangle", "square", "circle")
            shape_params: 形状参数
            color: 绘制颜色，默认绿色
            thickness: 线条粗细，默认2
        Returns:
            绘制后的图像
        """
        if shape_params is None:
            return frame
        
        result_frame = frame.copy()
        
        if shape_type == "circle":
            center = shape_params["center"]
            radius = shape_params["radius"]
            cv2.circle(result_frame, center, radius, color, thickness)
        elif shape_type in ["triangle", "square"]:
            # 绘制多边形
            points = shape_params.astype(np.int32)
            cv2.polylines(result_frame, [points], True, color, thickness)
        
        return result_frame
    
    def calculate_X(self, x_pix, D, K, a4_corners=None):
        """
        计算实际尺寸 - 综合三种方法
        Args:
            x_pix: 像素尺寸
            D: 距离值 (cm)
            K: 相机内参矩阵
            a4_corners: A4纸的四个角点 (4x2)
        Returns:
            x: 实际尺寸 (cm)
        """
        # A4纸黑边内边缘尺寸 (cm)
        A4_WIDTH = 17.0
        A4_HEIGHT = 25.7
        
        results = []
        weights = []
        
        # 方法1: 内参计算公式
        x1 = x_pix * D / K[1, 1]
        results.append(x1)
        weights.append(1.0)  # 基础权重
        
        if a4_corners is not None:
            # 计算A4纸的像素宽度和高度
            # corners顺序为: 左上, 右上, 右下, 左下
            width_top = np.linalg.norm(a4_corners[1] - a4_corners[0])
            width_bottom = np.linalg.norm(a4_corners[2] - a4_corners[3])
            height_left = np.linalg.norm(a4_corners[3] - a4_corners[0])
            height_right = np.linalg.norm(a4_corners[2] - a4_corners[1])
            
            # 取平均值
            a4_width_pix = (width_top + width_bottom) / 2
            a4_height_pix = (height_left + height_right) / 2
            
            # 方法2: 基于A4纸宽度的比例计算
            if a4_width_pix > 0:
                x2 = x_pix * A4_WIDTH / a4_width_pix
                results.append(x2)
                weights.append(1.5)  # A4参考通常更准确
            
            # 方法3: 基于A4纸高度的比例计算
            if a4_height_pix > 0:
                x3 = x_pix * A4_HEIGHT / a4_height_pix
                results.append(x3)
                weights.append(1.5)  # A4参考通常更准确
        
        # 加权平均计算最终结果
        if len(results) > 1:
            weighted_sum = sum(r * w for r, w in zip(results, weights))
            total_weight = sum(weights)
            final_result = weighted_sum / total_weight
            
            # 可选：输出调试信息
            print(f"尺寸计算结果: 内参法={results[0]:.2f}cm", end="")
            if len(results) > 1:
                print(f", A4宽度法={results[1]:.2f}cm", end="")
            if len(results) > 2:
                print(f", A4高度法={results[2]:.2f}cm", end="")
            print(f" -> 综合结果={final_result:.2f}cm")
            
            return final_result
        else:
            return results[0]