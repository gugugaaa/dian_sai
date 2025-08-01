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
        
        # 形状判断：仅基于多边形近似点数，不使用霍夫直线结果
        if num_sides == 3:
            shape = "triangle"
        elif num_sides == 4:
            shape = "square"
        else:
            return None, None
        
        # 步骤4：如果可能为多边形，计算Hough直线的交点（仅用于优化顶点坐标）
        hough_points = None
        if num_hough_lines == num_sides and lines is not None:
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
        
        # 步骤5：如有hough_points且数量匹配，则与approx点融合；否则直接使用approx
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
            print("仅使用approx点")
            shape_params = approx_points
        
        return shape, shape_params.astype(int)  # 返回int类型以保持一致性

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
            return "circle", shape_params.astype(int)

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
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
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
    

    def sort_corners(self, points):
        """
        对正方形的四个角点进行排序：TL, TR, BR, BL (顺时针)
        使用 x+y 和 x-y 方法排序
        """
        points = np.array(points, dtype=float)
        if len(points) != 4:
            raise ValueError("必须提供四个角点")
        
        # 计算 x + y 和 x - y
        sum_xy = points[:, 0] + points[:, 1]
        diff_xy = points[:, 0] - points[:, 1]
        
        # TL: 最小 sum_xy
        tl_idx = np.argmin(sum_xy)
        tl = points[tl_idx]
        
        # BR: 最大 sum_xy
        br_idx = np.argmax(sum_xy)
        br = points[br_idx]
        
        # TR: 最小 diff_xy (对于右上，x大 y小，x-y大? 等一下标准是:
        # 实际: 对于标准坐标 (y向下增加)
        # TL: min sum
        # BR: max sum
        # BL: min diff (x小 y大, x-y小)
        # TR: max diff (x大 y小, x-y大)
        
        # 修正:
        diff_xy = points[:, 0] - points[:, 1]
        bl_idx = np.argmin(diff_xy)
        bl = points[bl_idx]
        
        tr_idx = np.argmax(diff_xy)
        tr = points[tr_idx]
        
        # 验证是否四个独特点
        indices = {tl_idx, tr_idx, br_idx, bl_idx}
        if len(indices) != 4:
            # 如果重叠，使用中心和角度排序作为备用
            center = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_points = points[sorted_indices]
            # 假设逆时针，从最小角度开始
            tl, bl, br, tr = sorted_points  # 需根据实际调整
        else:
            sorted_points = np.array([tl, tr, br, bl])
        
        return sorted_points.astype(int)


    def rotate_and_crop(self, image, corners):
        """
        以图形底边为基准计算出倾斜角度，将图像和角点坐标同步旋转一个“反向”的角度，
        最后在摆正后的图像上根据新的角点坐标进行矩形裁切。
        """
        if len(corners) != 4:
            raise ValueError("必须提供四个角点")
        
        # 调用sort_corners确保角点顺序为TL, TR, BR, BL
        corners = self.sort_corners(corners)
        
        tl, tr, br, bl = corners
        
        # 计算底边 (BL to BR) 的角度
        dx = br[0] - bl[0]
        dy = br[1] - bl[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 假设当前底边是 y 较大的一边，左边是 x 较小的一边
        
        # 计算旋转中心：正方形中心
        center = np.mean(corners, axis=0).astype(int)
        
        # 计算需要旋转的角度，使底边水平 (angle to 0)
        rotation_angle = angle  # ！！！注意反向不是-angle
        
        # 计算padding后的新尺寸，以避免裁切丢失
        h, w = image.shape[:2]
        diag = np.sqrt(h**2 + w**2)
        new_size = int(np.ceil(diag))
        pad_h = (new_size - h) // 2
        pad_w = (new_size - w) // 2
        
        # padding图像
        padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # 更新中心
        new_center = (int(center[0] + pad_w), int(center[1] + pad_h))
        
        # 获取旋转矩阵
        rot_matrix = cv2.getRotationMatrix2D(new_center, rotation_angle, 1.0)
        
        # 旋转 padded_image
        rotated_image = cv2.warpAffine(padded_image, rot_matrix, (new_size, new_size), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        
        # 更新角点位置
        # 添加齐次坐标
        corners_hom = np.hstack([corners, np.ones((4, 1))])
        updated_corners_hom = np.dot(rot_matrix, corners_hom.T).T
        
        # 但由于用 new_center 旋转，角点需先加 padding
        orig_corners = np.array(corners)
        padded_corners = orig_corners + [pad_w, pad_h]
        padded_corners_hom = np.hstack([padded_corners, np.ones((4, 1))])
        rotated_corners = np.dot(rot_matrix, padded_corners_hom.T).T[:, :2].astype(int)
        
        # 现在底边应水平，BL 和 BR y 相同，且在底下 (最大 y)
        # 裁切：计算 min_x, min_y, width, height
        min_x = min(rotated_corners[:, 0])
        max_x = max(rotated_corners[:, 0])
        min_y = min(rotated_corners[:, 1])
        max_y = max(rotated_corners[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 由于是正方形，width ≈ height
        side = max(width, height)  # 取较大以安全
        
        cropped = rotated_image[min_y:min_y + side, min_x:min_x + side]
        
        return cropped

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
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
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
    
    def detect_squares_debug(self, frame, min_area=1000):
        """
        检测图像中多个不重叠的正方形 - 调试版本
        显示关键处理步骤
        Args:
            frame: 输入图像 (BGR)
            min_area: 最小轮廓面积阈值，用于过滤噪声 (默认1000)
        Returns:
            list of dict: 每个字典包含 {'shape': 'square', 'params': array of points (Nx2)}
            如果没有检测到，返回空列表 []
        """
        print("=== 开始多正方形检测调试 ===")
        
        # 显示原图
        cv2.imshow("Debug - Original Image", frame)
        cv2.waitKey(500)
        
        # 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Debug - Grayscale", gray)
        cv2.waitKey(500)
        
        # 二值化
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("Debug - Binary Threshold", thresh)
        cv2.waitKey(500)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未找到轮廓")
            cv2.destroyAllWindows()
            return []
        
        # 绘制所有轮廓
        all_contours_img = frame.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 255), 2)
        cv2.imshow("Debug - All Contours", all_contours_img)
        cv2.waitKey(500)
        
        # 过滤面积太小的轮廓
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        print(f"原始轮廓数: {len(contours)}, 过滤后轮廓数: {len(filtered_contours)}")
        
        if filtered_contours:
            filtered_contours_img = frame.copy()
            cv2.drawContours(filtered_contours_img, filtered_contours, -1, (0, 255, 0), 2)
            cv2.imshow("Debug - Filtered Contours", filtered_contours_img)
            cv2.waitKey(500)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imshow("Debug - Blurred", blurred)
        cv2.waitKey(500)
        
        detected_squares = []
        
        for idx, contour in enumerate(filtered_contours):
            print(f"\n--- 处理轮廓 {idx + 1}/{len(filtered_contours)} ---")
            area = cv2.contourArea(contour)
            print(f"轮廓面积: {area:.0f}")
            
            # 显示当前处理的轮廓
            current_contour_img = frame.copy()
            cv2.drawContours(current_contour_img, [contour], -1, (255, 0, 0), 3)
            cv2.imshow(f"Debug - Processing Contour {idx + 1}", current_contour_img)
            cv2.waitKey(500)
            
            # 多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            print(f"多边形近似点数: {len(approx)}")
            
            # 显示近似多边形
            approx_img = frame.copy()
            cv2.polylines(approx_img, [approx], True, (0, 0, 255), 2)
            # 标记顶点
            for i, point in enumerate(approx):
                cv2.circle(approx_img, tuple(point[0]), 5, (255, 0, 255), -1)
                cv2.putText(approx_img, str(i), tuple(point[0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(f"Debug - Approx Polygon {idx + 1}", approx_img)
            cv2.waitKey(500)
            
            # 创建ROI用于边缘检测
            N = 10
            x, y, w, h = cv2.boundingRect(approx)
            roi_x = max(0, x - N)
            roi_y = max(0, y - N)
            roi_w = min(gray.shape[1] - roi_x, w + 2 * N)
            roi_h = min(gray.shape[0] - roi_y, h + 2 * N)
            
            # 显示ROI区域
            roi_img = frame.copy()
            cv2.rectangle(roi_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
            cv2.imshow(f"Debug - ROI {idx + 1}", roi_img)
            cv2.waitKey(500)
            
            # 提取ROI并进行边缘检测
            roi_blurred = blurred[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            edges = cv2.Canny(roi_blurred, 50, 150)
            cv2.imshow(f"Debug - Canny Edges {idx + 1}", edges)
            cv2.waitKey(500)
            
            # Hough直线检测
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
            if lines is not None:
                print(f"检测到 {len(lines)} 条Hough直线")
                
                # 显示Hough直线
                hough_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.imshow(f"Debug - Hough Lines {idx + 1}", hough_img)
                cv2.waitKey(500)
            else:
                print("未检测到Hough直线")
            
            # 调用形状检测
            shape, shape_params = self._detect_polygonal(approx, contour, blurred, gray)
            print(f"检测结果: {shape}")
            
            if shape == "square":
                print("✓ 检测为正方形")
                detected_squares.append({'shape': 'square', 'params': shape_params})
                
                # 显示检测到的正方形
                square_img = frame.copy()
                points = shape_params.astype(np.int32)
                cv2.polylines(square_img, [points], True, (0, 255, 0), 3)
                # 标记顶点
                for i, point in enumerate(points):
                    cv2.circle(square_img, tuple(point), 5, (0, 255, 0), -1)
                    cv2.putText(square_img, str(i), tuple(point + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(f"Debug - Detected Square {len(detected_squares)}", square_img)
                cv2.waitKey(1000)
            else:
                print("✗ 非正方形")
        
        # 显示最终结果
        if detected_squares:
            final_result = frame.copy()
            for i, square in enumerate(detected_squares):
                points = square['params'].astype(np.int32)
                cv2.polylines(final_result, [points], True, (0, 255, 0), 3)
                # 在中心标记序号
                center = np.mean(points, axis=0).astype(int)
                cv2.putText(final_result, str(i+1), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Debug - Final Result - All Squares", final_result)
            print(f"\n=== 检测完成，共找到 {len(detected_squares)} 个正方形 ===")
        else:
            cv2.imshow("Debug - Final Result - No Squares", frame)
            print("\n=== 检测完成，未找到正方形 ===")
        
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        return detected_squares