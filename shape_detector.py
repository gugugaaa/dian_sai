# shape_detector.py (modified)
import cv2
import numpy as np

class ShapeDetector:
    def _detect_polygonal(self, approx, target_contour, blurred, gray):
        # 说明：增加 target_contour, blurred, gray 作为输入参数用于ROI处理
        num_sides = len(approx)
        print("num_sides:", num_sides)
        
        # 步骤1：基于approx或轮廓创建扩展的ROI区域
        # 这里简单用approx的外接矩形，并向外扩展N像素
        N = 10  # 扩展像素，可调节（建议10-20）
        x, y, w, h = cv2.boundingRect(approx)
        # 扩展并裁剪到图像边界
        roi_x = max(0, x - N)
        roi_y = max(0, y - N)
        roi_w = min(gray.shape[1] - roi_x, w + 2 * N)
        roi_h = min(gray.shape[0] - roi_y, h + 2 * N)
        
        # 从模糊图像中提取ROI区域（用于Canny），如有需要也可从灰度图提取
        roi_blurred = blurred[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # 步骤2：在局部ROI上做Canny边缘检测
        edges = cv2.Canny(roi_blurred, 50, 150)
        
        # 步骤3：在局部边缘图上做Hough直线检测
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
        
        # 步骤4：如果可能为多边形，计算Hough直线的交点
        hough_points = None
        if num_hough_lines in [3, 4] and lines is not None:
            # 将直线转换为(rho, theta)并计算交点
            selected_lines = []  # 每组选择一条代表性直线
            for group in groups:
                # 取每组的中位数theta
                median_theta = np.median(group)
                # 找到与中位数theta最接近的直线
                closest_idx = np.argmin(np.abs(thetas - median_theta))
                selected_lines.append(lines[closest_idx, 0])  # [rho, theta]
            
            if len(selected_lines) == num_hough_lines:
                hough_points = self._compute_line_intersections(selected_lines, (roi_x, roi_y))
        
        # 形状判断（保留或条件）
        if num_sides == 3 or num_hough_lines == 3:
            shape = "triangle"
        elif num_sides == 4 or num_hough_lines == 4:
            shape = "square"
        else:
            return None, None
        
        # 步骤5：如有hough_points，则与approx点融合
        approx_points = approx.reshape(-1, 2).astype(float)
        if hough_points is not None and len(hough_points) == num_sides:
            # 简单融合：按最近邻匹配并取均值
            # 假设点顺序相近，若需更鲁棒可按角度排序或匈牙利匹配
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(approx_points, hough_points)
            matches = np.argmin(dist_matrix, axis=1)
            fused_points = (approx_points + hough_points[matches]) / 2.0
            # 筛选：如有匹配距离过大（>50像素），则退回approx
            if np.max(np.min(dist_matrix, axis=1)) < 50:
                # 打印调试信息：每对匹配点的距离和坐标
                print("匹配成功的点对及其距离：")
                for i in range(len(approx_points)):
                    idx = matches[i]
                    dist = dist_matrix[i, idx]
                    print(f"approx点{i}: {approx_points[i]}, hough点{idx}: {hough_points[idx]}, 距离: {dist:.2f}")
                shape_params = fused_points
            else:
                # 打印未匹配成功时每个点的最小距离
                print("未匹配成功，approx点与最近hough点的最小距离：")
                for i in range(len(approx_points)):
                    min_dist = np.min(dist_matrix[i])
                    min_idx = np.argmin(dist_matrix[i])
                    print(f"approx点{i}: {approx_points[i]}, 最近hough点{min_idx}: {hough_points[min_idx]}, 距离: {min_dist:.2f}")
                shape_params = approx_points
        else:
            shape_params = approx_points
        
        return shape, shape_params.astype(int)  # 返回int类型以保持一致性

    def _compute_line_intersections(self, lines, roi_offset):
        # 计算所有直线两两的交点（输入为rho, theta）
        # 对于三角形/正方形，期望有3/4个交点
        points = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i]
                rho2, theta2 = lines[j]
                # 跳过近似平行的直线
                if abs(theta1 - theta2) < 0.1 or abs(theta1 - theta2) > np.pi - 0.1:
                    continue
                # 计算交点
                a1 = np.cos(theta1)
                b1 = np.sin(theta1)
                a2 = np.cos(theta2)
                b2 = np.sin(theta2)
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-6:
                    continue
                x = (b2 * rho1 - b1 * rho2) / det
                y = (a1 * rho2 - a2 * rho1) / det
                # 加上ROI偏移，得到全局坐标
                points.append([x + roi_offset[0], y + roi_offset[1]])
        
        # 简单去重：如果两个点距离小于5像素，则合并为均值
        if len(points) > 0:
            points = np.array(points)
            unique_points = []
            used = np.zeros(len(points), dtype=bool)
            for i in range(len(points)):
                if used[i]:
                    continue
                close = [points[i]]
                used[i] = True
                for j in range(i + 1, len(points)):
                    if not used[j] and np.linalg.norm(points[i] - points[j]) < 5:
                        close.append(points[j])
                        used[j] = True
                unique_points.append(np.mean(close, axis=0))
            if len(unique_points) == len(lines):  # 期望点数等于直线数
                return np.array(unique_points)
        
        return None

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
        # 传递额外参数给_detect_polygonal
        shape, shape_params = self._detect_polygonal(approx, target_contour, blurred, gray)
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
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                side = np.linalg.norm(p1 - p2)
                sides.append(side)
            x_pix = np.mean(sides)

        return shape, x_pix, shape_params
    
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
        weights.append(0.8)  # 基础权重
        
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
    
    def detect_squares(self, frame, min_area=1000):
        """
        检测图像中多个不重叠的正方形。
        Args:
            frame: 输入图像 (BGR)
            min_area: 最小轮廓面积阈值，用于过滤噪声 (默认1000)
        Returns:
            list of dict: 每个字典包含 {'shape': 'square', 'params': array of points (Nx2)}
            如果没有检测到，返回空列表 []
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        detected_squares = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue  # 跳过太小的轮廓
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            shape, shape_params = self._detect_polygonal(approx, contour, blurred, gray)
            
            if shape == "square":
                detected_squares.append({'shape': 'square', 'params': shape_params})
                
        
        return detected_squares