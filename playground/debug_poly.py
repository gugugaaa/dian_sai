import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import cv2
import numpy as np
import math
from itertools import combinations
from system_initializer import MeasurementSystem


class IntegratedPolygonSquareDetector:
    """
    集成多边形检测和正方形构造算法的类
    """
    
    def __init__(self, 
                 gaussian_blur_kernel=(5, 5),
                 canny_low_threshold=50,
                 canny_high_threshold=150,
                 min_perimeter=50,
                 outer_epsilon_factor=0.005,
                 inner_epsilon_factor=0.005,
                 outer_vertex_merge_threshold=8.0,
                 inner_vertex_merge_threshold=6.0,
                 similarity_perimeter_threshold=0.1,
                 similarity_vertex_threshold=2):
        """
        初始化检测器
        """
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.min_perimeter = min_perimeter
        self.outer_epsilon_factor = outer_epsilon_factor
        self.inner_epsilon_factor = inner_epsilon_factor
        self.outer_vertex_merge_threshold = outer_vertex_merge_threshold
        self.inner_vertex_merge_threshold = inner_vertex_merge_threshold
        self.similarity_perimeter_threshold = similarity_perimeter_threshold
        self.similarity_vertex_threshold = similarity_vertex_threshold
        
        self.outer_polygons = []
        self.inner_polygons = []
        self.image_shape = None
        
        # 添加调试图像保存路径 - 为不同阶段创建不同文件夹
        self.debug_base_dir = "debug_images"
        self.debug_dirs = {
            'detected_vertices': os.path.join(self.debug_base_dir, 'stage0_detected_vertices'),
            'non_convex_vertices': os.path.join(self.debug_base_dir, 'stage1_non_convex_vertices'),
            'non_90_degree_vertices': os.path.join(self.debug_base_dir, 'stage2_non_90_degree_vertices'),
            'accepted_vertices': os.path.join(self.debug_base_dir, 'stage2_accepted_vertices'),  # 新增
            'line_outside_shape': os.path.join(self.debug_base_dir, 'stage3_line_outside_shape'),
            'squares_outside_polygon': os.path.join(self.debug_base_dir, 'stage4_squares_outside_polygon'),
            'accepted_combinations': os.path.join(self.debug_base_dir, 'stage5_accepted_combinations')
        }
        
        # 创建所有调试文件夹
        for debug_dir in self.debug_dirs.values():
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
        
        self.debug_frame = None  # 保存当前帧用于调试
        self.debug_counters = {key: 0 for key in self.debug_dirs.keys()}  # 为每个阶段单独计数
    
    def preprocess_image(self, frame):
        """图像预处理：灰度化 -> 高斯模糊 -> OTSU阈值 -> Canny边缘检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_kernel, 0)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(otsu_thresh, self.canny_low_threshold, self.canny_high_threshold)
        return edges
    
    def find_contours_with_hierarchy(self, edges):
        """查找轮廓并获取层次结构信息"""
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    def is_similar_contour(self, poly1, poly2):
        """判断两个轮廓是否相似（可能是同一形状的内外边缘）"""
        perimeter_ratio = abs(poly1['perimeter'] - poly2['perimeter']) / max(poly1['perimeter'], poly2['perimeter'])
        vertex_diff = abs(poly1['vertices'] - poly2['vertices'])
        return perimeter_ratio < self.similarity_perimeter_threshold and vertex_diff <= self.similarity_vertex_threshold
    
    def process_contours(self, contours, hierarchy):
        """处理轮廓：分类、拟合、过滤重复轮廓、合并邻近顶点"""
        all_polygons = []
        outer_polygons = []
        inner_polygons = []
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            
            for i, contour in enumerate(contours):
                perimeter = cv2.arcLength(contour, True)
                if perimeter < self.min_perimeter:
                    continue
                
                next_contour, prev_contour, first_child, parent = hierarchy[i]
                is_outer = parent == -1
                
                epsilon = self.outer_epsilon_factor * perimeter if is_outer else self.inner_epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # +++ START: 添加顶点合并逻辑 +++
                # 根据是内/外轮廓，选择不同的合并阈值
                merge_threshold = self.outer_vertex_merge_threshold if is_outer else self.inner_vertex_merge_threshold
                
                # 对当前多边形的顶点进行合并
                merged_approx = self.merge_polygon_vertices(approx, distance_threshold=merge_threshold)
                # +++ END: 添加顶点合并逻辑 +++

                # 使用合并后的顶点重新计算面积，虽然不必须，但更准确
                area = cv2.contourArea(merged_approx)
                
                polygon_info = {
                    'contour': contour,
                    'approx': merged_approx,  # <-- 使用合并后的顶点
                    'vertices': len(merged_approx), # <-- 使用合并后的顶点数量
                    'perimeter': perimeter,
                    'area': area,
                    'id': i,
                    'parent': parent,
                    'has_children': first_child != -1,
                    'is_outer': is_outer
                }
                
                all_polygons.append(polygon_info)
            
            # 后续逻辑不变
            for poly in all_polygons:
                if poly['is_outer']:
                    outer_polygons.append(poly)
                else:
                    parent_id = poly['parent']
                    parent_poly = None
                    
                    for p in all_polygons:
                        if p['id'] == parent_id:
                            parent_poly = p
                            break
                    
                    if parent_poly and self.is_similar_contour(poly, parent_poly):
                        continue
                    else:
                        inner_polygons.append(poly)
        
        return all_polygons, outer_polygons, inner_polygons
    
    
    def merge_polygon_vertices(self, vertices, distance_threshold=5.0):
        """
        合并位置差异太小的顶点，保持原有轮廓的顺序
        """
        if len(vertices) <= 2:
            return vertices
        
        # 提取顶点坐标，保持原有顺序
        points = np.array([vertex[0] for vertex in vertices], dtype=np.float32)
        n_points = len(points)
        
        if n_points == 0:
            return vertices
        
        # 用于标记哪些顶点需要被合并
        to_merge = []
        merged_indices = set()
        
        # 检查相邻顶点之间的距离（考虑轮廓的循环性质）
        i = 0
        while i < n_points:
            if i in merged_indices:
                i += 1
                continue
                
            current_point = points[i]
            merge_group = [i]  # 当前合并组，包含当前顶点索引
            
            # 向前查找相邻的近距离顶点
            j = (i + 1) % n_points
            consecutive_count = 0  # 连续近距离顶点的计数
            
            while j != i and consecutive_count < n_points - 1:
                if j in merged_indices:
                    j = (j + 1) % n_points
                    consecutive_count += 1
                    continue
                    
                distance = np.linalg.norm(points[j] - current_point)
                if distance < distance_threshold:
                    merge_group.append(j)
                    merged_indices.add(j)
                    j = (j + 1) % n_points
                    consecutive_count += 1
                else:
                    break
            
            # 向后查找相邻的近距离顶点（仅在闭合轮廓的情况下）
            if len(merge_group) > 1:  # 如果已经找到了需要合并的顶点
                j = (i - 1) % n_points
                consecutive_count = 0
                
                while j != i and consecutive_count < n_points - 1 and j not in merged_indices:
                    distance = np.linalg.norm(points[j] - current_point)
                    if distance < distance_threshold:
                        merge_group.insert(0, j)  # 插入到前面保持顺序
                        merged_indices.add(j)
                        j = (j - 1) % n_points
                        consecutive_count += 1
                    else:
                        break
            
            to_merge.append(merge_group)
            merged_indices.add(i)
            i += 1
        
        # 构建合并后的顶点列表，保持原有顺序
        merged_vertices = []
        processed_indices = set()
        
        for i in range(n_points):
            if i in processed_indices:
                continue
                
            # 找到包含当前索引的合并组
            merge_group = None
            for group in to_merge:
                if i in group:
                    merge_group = group
                    break
            
            if merge_group and len(merge_group) > 1:
                # 计算合并组的平均位置
                group_points = points[merge_group]
                merged_point = np.mean(group_points, axis=0)
                merged_vertices.append(merged_point)
                
                # 标记这些索引为已处理
                for idx in merge_group:
                    processed_indices.add(idx)
            else:
                # 保持原有顶点
                merged_vertices.append(points[i])
                processed_indices.add(i)
        
        # 转换回OpenCV格式
        if len(merged_vertices) == 0:
            return vertices
            
        result = np.array([[[int(point[0]), int(point[1])]] for point in merged_vertices])
        return result

    def detect_polygons(self, frame):
        """主要的多边形检测方法"""
        self.image_shape = frame.shape
        self.debug_frame = frame.copy()  # 保存帧用于调试
        edges = self.preprocess_image(frame)
        contours, hierarchy = self.find_contours_with_hierarchy(edges)
        all_polygons, outer_polygons, inner_polygons = self.process_contours(contours, hierarchy)

        self.outer_polygons = outer_polygons
        self.inner_polygons = inner_polygons
        
        # 保存检测到的所有顶点
        self._save_detected_vertices()
        
        return outer_polygons, inner_polygons
    
    def _get_point_coordinate(self, point):
        """获取点的坐标，支持多种输入格式"""
        if isinstance(point, (tuple, list)) and len(point) == 2:
            try:
                return (float(point[0]), float(point[1]))
            except (ValueError, TypeError):
                return None
        return None
    
    def is_line_inside_shape(self, point1, point2):
        """判断两个点之间的连线是否完全在图形内部"""
        p1 = self._get_point_coordinate(point1)
        p2 = self._get_point_coordinate(point2)
        
        if p1 is None or p2 is None:
            return False
        
        tolerance = 3.0
        
        # 检查是否冲破外轮廓
        for poly in self.outer_polygons:
            if self._line_breaks_through_outer_contour(p1, p2, poly['approx'], tolerance):
                return False
        
        # 检查是否冲破内轮廓  
        for poly in self.inner_polygons:
            if self._line_breaks_through_inner_contour(p1, p2, poly['approx'], tolerance):
                return False
        
        # 确保至少一个端点在有效区域内
        if not (self._point_in_valid_area_with_tolerance(p1, tolerance) or 
                self._point_in_valid_area_with_tolerance(p2, tolerance)):
            return False
        
        # 检查线段上的采样点（1/4、1/2、3/4）是否都在有效区域内
        sample_points = [
            (p1[0] + 0.25 * (p2[0] - p1[0]), p1[1] + 0.25 * (p2[1] - p1[1])),  # 1/4点
            (p1[0] + 0.5 * (p2[0] - p1[0]), p1[1] + 0.5 * (p2[1] - p1[1])),    # 1/2点
            (p1[0] + 0.75 * (p2[0] - p1[0]), p1[1] + 0.75 * (p2[1] - p1[1]))   # 3/4点
        ]
        
        for sample_point in sample_points:
            if not self._point_in_valid_area_with_tolerance(sample_point, tolerance):
                return False
        
        return True

    def _point_in_valid_area_with_tolerance(self, point, tolerance=3.0):
        """检查点是否在有效区域内，带容忍度处理"""
        in_outer = False
        for poly in self.outer_polygons:
            distance = cv2.pointPolygonTest(poly['approx'], point, True)
            if distance >= -tolerance:
                in_outer = True
                break
        
        if not in_outer:
            return False
        
        for poly in self.inner_polygons:
            distance = cv2.pointPolygonTest(poly['approx'], point, True)
            if distance > tolerance:
                return False
        
        return True

    def _line_breaks_through_outer_contour(self, p1, p2, contour, tolerance=3.0):
        """判断线段是否冲破外轮廓"""
        p1_dist = cv2.pointPolygonTest(contour, p1, True)
        p2_dist = cv2.pointPolygonTest(contour, p2, True)
        
        p1_outside = p1_dist < -tolerance
        p2_outside = p2_dist < -tolerance
        
        if not (p1_outside or p2_outside):
            return False
        
        intersections = self._get_line_contour_intersections(p1, p2, contour)
        return len(intersections) > 0

    def _line_breaks_through_inner_contour(self, p1, p2, contour, tolerance=3.0):
        """判断线段是否冲破内轮廓"""
        intersections = self._get_line_contour_intersections(p1, p2, contour)
        
        if len(intersections) == 0:
            return False
        
        return self._line_crosses_inner_contour(p1, p2, contour, intersections, tolerance)

    def _line_crosses_inner_contour(self, p1, p2, contour, intersections, tolerance=3.0):
        """判断线段是否真正穿过内轮廓内部"""
        if len(intersections) >= 2:
            return True
        elif len(intersections) == 1:
            mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid_dist = cv2.pointPolygonTest(contour, mid_point, True)
            return mid_dist > tolerance
        else:
            return False

    def _get_line_contour_intersections(self, p1, p2, contour):
        """计算线段与轮廓的所有交点"""
        intersections = []
        contour_points = contour.reshape(-1, 2)
        n_points = len(contour_points)
        
        for i in range(n_points):
            edge_p1 = tuple(contour_points[i])
            edge_p2 = tuple(contour_points[(i + 1) % n_points])
            
            intersection = self._line_segments_intersect(p1, p2, edge_p1, edge_p2)
            if intersection is not None:
                intersections.append(intersection)
        
        return intersections

    def _line_segments_intersect(self, p1, p2, p3, p4):
        """计算两条线段的交点（如果存在）"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None

    def create_squares_from_segment(self, p1, p2):
        """
        根据线段端点创建三个正方形
        p1, p2: 线段的两个端点 (x, y)
        返回: 三个正方形的顶点列表
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # 计算线段向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 计算垂直向量（用于构造边正方形）
        perp_dx = -dy
        perp_dy = dx
        
        squares = []
        
        # 正方形1: 以线段为边，向一侧构造
        square1 = [
            (x1, y1),
            (x2, y2),
            (x2 + perp_dx, y2 + perp_dy),
            (x1 + perp_dx, y1 + perp_dy)
        ]
        squares.append(square1)
        
        # 正方形2: 以线段为边，向另一侧构造
        square2 = [
            (x1, y1),
            (x2, y2),
            (x2 - perp_dx, y2 - perp_dy),
            (x1 - perp_dx, y1 - perp_dy)
        ]
        squares.append(square2)
        
        # 正方形3: 以线段为对角线构造
        # 对角线的中点
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # 对角线长度的一半
        half_diag = math.sqrt(dx*dx + dy*dy) / 2
        
        # 计算另外两个顶点（垂直于对角线方向）
        angle = math.atan2(dy, dx)
        perp_angle = angle + math.pi / 2
        
        offset_x = half_diag * math.cos(perp_angle)
        offset_y = half_diag * math.sin(perp_angle)
        
        square3 = [
            (x1, y1),
            (mid_x + offset_x, mid_y + offset_y),
            (x2, y2),
            (mid_x - offset_x, mid_y - offset_y)
        ]
        squares.append(square3)
        
        return squares
    
    def point_in_polygon_with_tolerance(self, point, polygon_contour, tolerance=5.0):
        """
        检查点是否在多边形内部，带容忍度处理
        """
        # 将轮廓转换为正确的格式
        if isinstance(polygon_contour, list):
            contour = np.array([(int(x), int(y)) for x, y in polygon_contour], dtype=np.int32)
        else:
            contour = polygon_contour
        
        # 使用cv2.pointPolygonTest计算点到多边形的距离
        distance = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
        
        # 如果距离大于等于负容忍度，则认为点在多边形内部
        return distance >= -tolerance
    
    def square_in_polygon(self, square, polygon_contour, tolerance=5.0):
        """
        检测正方形是否在多边形内部
        算法：检查正方形的四个角是否都在多边形内部（包括边缘）
        """
        corner_results = []
        all_inside = True
        
        for i, corner in enumerate(square):
            is_inside = self.point_in_polygon_with_tolerance(corner, polygon_contour, tolerance)
            corner_results.append({
                'corner_index': i,
                'position': corner,
                'is_inside': is_inside
            })
            if not is_inside:
                all_inside = False
        
        return all_inside, corner_results
    
    def calculate_square_side_length(self, p1, p2):
        """计算由两点构成的正方形的边长"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def is_convex_vertex(self, contour_type, contour_id, vertex_id, step_distance=10.0):
        """
        检测指定顶点是否是凸点
        
        Args:
            contour_type: 'outer' 或 'inner'
            contour_id: 轮廓序号
            vertex_id: 顶点序号
            step_distance: 沿内角平分线走的步数，默认10像素
        
        Returns:
            bool: True if convex, False otherwise
        """
        try:
            if contour_type == 'outer':
                if contour_id >= len(self.outer_polygons):
                    return False
                vertices = self.outer_polygons[contour_id]['approx']
            elif contour_type == 'inner':
                if contour_id >= len(self.inner_polygons):
                    return False
                vertices = self.inner_polygons[contour_id]['approx']
            else:
                return False
            
            n_vertices = len(vertices)
            if n_vertices < 3 or vertex_id >= n_vertices:
                return False
            
            prev_idx = (vertex_id - 1) % n_vertices
            curr_idx = vertex_id
            next_idx = (vertex_id + 1) % n_vertices
            
            prev_point = vertices[prev_idx][0].astype(float)
            curr_point = vertices[curr_idx][0].astype(float)
            next_point = vertices[next_idx][0].astype(float)
            
            # 计算两条边的向量
            vector1 = prev_point - curr_point  # 从当前点指向前一个点
            vector2 = next_point - curr_point  # 从当前点指向下一个点
            
            # 归一化向量
            len1 = np.linalg.norm(vector1)
            len2 = np.linalg.norm(vector2)
            
            if len1 == 0 or len2 == 0:
                return False
            
            vector1_norm = vector1 / len1
            vector2_norm = vector2 / len2
            
            # 计算内角平分线方向（两个归一化向量的和）
            bisector = vector1_norm + vector2_norm
            bisector_len = np.linalg.norm(bisector)
            
            if bisector_len == 0:
                # 如果两个向量相反（180度角），则不是凸点
                return False
            
            bisector_norm = bisector / bisector_len
            
            # 沿内角平分线方向走step_distance步
            test_point = curr_point + bisector_norm * step_distance
            
            # 检查这个点是否在外边框内
            return self._point_in_valid_area_with_tolerance(tuple(test_point), tolerance=3.0)
            
        except Exception:
            return False
    
    def get_vertex_angle(self, contour_type, contour_id, vertex_id):
        """
        计算指定顶点的两条连接边的夹角
        
        Args:
            contour_type: 'outer' 或 'inner'
            contour_id: 轮廓序号
            vertex_id: 顶点序号
        
        Returns:
            float: 夹角度数（0-180度），如果无效则返回None
        """
        try:
            if contour_type == 'outer':
                if contour_id >= len(self.outer_polygons):
                    return None
                vertices = self.outer_polygons[contour_id]['approx']
            elif contour_type == 'inner':
                if contour_id >= len(self.inner_polygons):
                    return None
                vertices = self.inner_polygons[contour_id]['approx']
            else:
                return None
            
            n_vertices = len(vertices)
            if n_vertices < 3 or vertex_id >= n_vertices:
                return None
            
            prev_idx = (vertex_id - 1) % n_vertices
            curr_idx = vertex_id
            next_idx = (vertex_id + 1) % n_vertices
            
            prev_point = vertices[prev_idx][0].astype(float)
            curr_point = vertices[curr_idx][0].astype(float)
            next_point = vertices[next_idx][0].astype(float)
            
            vector1 = prev_point - curr_point
            vector2 = next_point - curr_point
            
            len1 = np.linalg.norm(vector1)
            len2 = np.linalg.norm(vector2)
            
            if len1 == 0 or len2 == 0:
                return None
            
            dot_product = np.dot(vector1, vector2)
            cos_angle = dot_product / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception:
            return None
    
    def _save_accepted_vertex(self, stage, coordinates, contour_id, vertex_id, reason):
        """
        保存被接受的单个顶点调试图像
        
        Args:
            stage: 阶段名称
            coordinates: 顶点坐标
            contour_id: 轮廓ID
            vertex_id: 顶点ID
            reason: 接受原因
        """
        if self.debug_frame is None:
            return
        
        # 创建调试图像
        debug_img = self.debug_frame.copy()
        
        # 绘制外轮廓（绿色）
        for poly in self.outer_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 255, 0), 2)
        
        # 绘制内轮廓（红色）
        for poly in self.inner_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 0, 255), 2)
        
        # 突出显示被接受的顶点（绿色大圆点）
        cv2.circle(debug_img, (int(coordinates[0]), int(coordinates[1])), 10, (0, 255, 0), -1)
        
        # 标记顶点编号
        cv2.putText(debug_img, f"C{contour_id}V{vertex_id}", 
                (int(coordinates[0])+15, int(coordinates[1])-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 添加接受原因
        cv2.putText(debug_img, f"Accept: {reason}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(debug_img, f"Vertex C{contour_id}V{vertex_id} at {coordinates}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 保存图像
        counter = self.debug_counters[stage]
        filename = os.path.join(self.debug_dirs[stage], f"{stage}_{counter:04d}.png")
        cv2.imwrite(filename, debug_img)
        print(f"保存调试图像: {filename}")
        
        # 递增计数器
        self.debug_counters[stage] += 1

    def get_90_degree_vertices(self, angle_tolerance=15):
        """
        获取外轮廓中所有接近90度角的凸顶点
        
        Args:
            angle_tolerance: 角度容忍度（度数），默认15度
            
        Returns:
            list: 有效顶点列表
        """
        valid_vertices = []
        
        for contour_id, poly in enumerate(self.outer_polygons):
            vertices = poly['approx']
            n_vertices = len(vertices)
            
            for vertex_id in range(n_vertices):
                coordinates = tuple(vertices[vertex_id][0])
                
                # 首先检查是否是凸点
                is_convex = self.is_convex_vertex('outer', contour_id, vertex_id)
                if not is_convex:
                    # 保存被拒绝的非凸顶点
                    self._save_debug_vertex('non_convex_vertices', coordinates, 
                                          contour_id, vertex_id, 'Non-convex vertex')
                    continue
                
                # 然后检查角度
                angle = self.get_vertex_angle('outer', contour_id, vertex_id)
                
                if angle is not None:
                    angle_diff = abs(angle - 90.0)
                    
                    if angle_diff <= angle_tolerance:
                        valid_vertices.append({
                            'contour_id': contour_id,
                            'vertex_id': vertex_id,
                            'angle': angle,
                            'coordinates': coordinates,
                            'is_convex': True  # 标记为凸点
                        })
                        # 保存被采纳的90度角顶点
                        self._save_accepted_vertex('accepted_vertices', coordinates,
                                                 contour_id, vertex_id,
                                                 f'Angle: {angle:.1f}° (diff: {angle_diff:.1f}°)')
                    else:
                        # 保存被拒绝的非90度顶点
                        self._save_debug_vertex('non_90_degree_vertices', coordinates,
                                              contour_id, vertex_id, 
                                              f'Angle: {angle:.1f}° (diff: {angle_diff:.1f}°)')
        
        return valid_vertices

    def find_minimum_square_points(self):
        """
        找到构造正方形边长最小的两个点
        """
        # 获取所有90度角凸顶点
        valid_vertices = self.get_90_degree_vertices(angle_tolerance=20)
        
        if len(valid_vertices) < 2:
            return None, None, float('inf'), []
        
        # 提取坐标用于组合
        vertex_coordinates = [vertex['coordinates'] for vertex in valid_vertices]
        
        # 遍历所有两点组合
        valid_combinations = []
        
        for p1, p2 in combinations(vertex_coordinates, 2):
            # 检查连线是否在图形内部
            if not self.is_line_inside_shape(p1, p2):
                # 保存被拒绝的线段（线段在形状外部）
                self._save_debug_line('line_outside_shape', p1, p2, 'Line outside shape')
                continue
            
            # 构造三个正方形
            squares = self.create_squares_from_segment(p1, p2)
            
            # 检查是否有正方形在外轮廓内部
            has_valid_square = False
            square_results = []
            
            for i, square in enumerate(squares):
                for poly in self.outer_polygons:
                    is_inside, corner_results = self.square_in_polygon(square, poly['approx'], tolerance=5.0)
                    square_results.append({
                        'square_index': i,
                        'polygon_index': poly['id'],
                        'is_inside': is_inside,
                        'corner_results': corner_results
                    })
                    
                    if is_inside:
                        has_valid_square = True
            
            # 如果有有效正方形，计算边长
            if has_valid_square:
                side_length = self.calculate_square_side_length(p1, p2)
                valid_combinations.append({
                    'point1': p1,
                    'point2': p2,
                    'side_length': side_length,
                    'squares': squares,
                    'square_results': square_results
                })
                # 保存被接受的组合
                self._save_debug_combination('accepted_combinations', p1, p2, squares, square_results, 
                                           f'Side length: {side_length:.2f}px')
            else:
                # 保存被拒绝的正方形（正方形在多边形外部）
                self._save_debug_combination('squares_outside_polygon', p1, p2, squares, square_results, 
                                           'All squares outside polygon')
        
        if not valid_combinations:
            return None, None, float('inf'), []
        
        # 找到边长最小的组合
        min_combination = min(valid_combinations, key=lambda x: x['side_length'])
        
        return (min_combination['point1'], 
                min_combination['point2'], 
                min_combination['side_length'], 
                valid_combinations)
    
    def _save_debug_vertex(self, stage, coordinates, contour_id, vertex_id, reason):
        """
        保存被拒绝的单个顶点调试图像
        
        Args:
            stage: 阶段名称
            coordinates: 顶点坐标
            contour_id: 轮廓ID
            vertex_id: 顶点ID
            reason: 拒绝原因
        """
        if self.debug_frame is None:
            return
        
        # 创建调试图像
        debug_img = self.debug_frame.copy()
        
        # 绘制外轮廓（绿色）
        for poly in self.outer_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 255, 0), 2)
        
        # 绘制内轮廓（红色）
        for poly in self.inner_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 0, 255), 2)
        
        # 突出显示被拒绝的顶点（红色大圆点）
        cv2.circle(debug_img, (int(coordinates[0]), int(coordinates[1])), 10, (0, 0, 255), -1)
        
        # 标记顶点编号
        cv2.putText(debug_img, f"C{contour_id}V{vertex_id}", 
                   (int(coordinates[0])+15, int(coordinates[1])-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 添加拒绝原因
        cv2.putText(debug_img, f"Reject: {reason}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(debug_img, f"Vertex C{contour_id}V{vertex_id} at {coordinates}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # 保存图像
        counter = self.debug_counters[stage]
        filename = os.path.join(self.debug_dirs[stage], f"{stage}_{counter:04d}.png")
        cv2.imwrite(filename, debug_img)
        print(f"保存调试图像: {filename}")
        
        # 递增计数器
        self.debug_counters[stage] += 1
    
    def _save_debug_line(self, stage, p1, p2, reason):
        """
        保存被拒绝的线段调试图像
        
        Args:
            stage: 阶段名称
            p1, p2: 线段端点
            reason: 拒绝原因
        """
        if self.debug_frame is None:
            return
        
        # 创建调试图像
        debug_img = self.debug_frame.copy()
        
        # 绘制外轮廓（绿色）
        for poly in self.outer_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 255, 0), 2)
        
        # 绘制内轮廓（红色）
        for poly in self.inner_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 0, 255), 2)
        
        # 绘制被拒绝的线段（红色粗线）
        cv2.line(debug_img, 
                (int(p1[0]), int(p1[1])), 
                (int(p2[0]), int(p2[1])), 
                (0, 0, 255), 3)
        
        # 绘制线段端点（红色圆点）
        cv2.circle(debug_img, (int(p1[0]), int(p1[1])), 6, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(p2[0]), int(p2[1])), 6, (0, 0, 255), -1)
        
        # 添加拒绝原因
        cv2.putText(debug_img, f"Reject: {reason}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(debug_img, f"Line: {p1} -> {p2}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # 保存图像
        counter = self.debug_counters[stage]
        filename = os.path.join(self.debug_dirs[stage], f"{stage}_{counter:04d}.png")
        cv2.imwrite(filename, debug_img)
        print(f"保存调试图像: {filename}")
        
        # 递增计数器
        self.debug_counters[stage] += 1
    
    def _save_debug_combination(self, stage, p1, p2, squares, square_results, reason):
        """
        保存组合调试图像（被拒绝的正方形或被接受的组合）
        
        Args:
            stage: 阶段名称
            p1, p2: 线段端点
            squares: 三个正方形列表
            square_results: 正方形检测结果
            reason: 原因（拒绝原因或接受信息）
        """
        if self.debug_frame is None:
            return
        
        # 创建调试图像
        debug_img = self.debug_frame.copy()
        
        # 绘制外轮廓（绿色）
        for poly in self.outer_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 255, 0), 2)
        
        # 绘制内轮廓（红色）
        for poly in self.inner_polygons:
            cv2.polylines(debug_img, [poly['approx']], True, (0, 0, 255), 2)
        
        # 选择线段和端点的颜色
        line_color = (255, 0, 255) if stage == 'accepted_combinations' else (0, 0, 255)  # 紫色 vs 红色
        
        # 绘制线段
        cv2.line(debug_img, 
                (int(p1[0]), int(p1[1])), 
                (int(p2[0]), int(p2[1])), 
                line_color, 3)
        
        # 绘制线段端点
        cv2.circle(debug_img, (int(p1[0]), int(p1[1])), 6, line_color, -1)
        cv2.circle(debug_img, (int(p2[0]), int(p2[1])), 6, line_color, -1)
        
        # 绘制三个正方形，不同颜色
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红、绿、蓝
        color_names = ['Red', 'Green', 'Blue']
        
        for i, square in enumerate(squares):
            points = np.array([(int(x), int(y)) for x, y in square], np.int32)
            
            # 检查这个正方形是否在多边形内部
            is_inside = False
            for result in square_results:
                if result['square_index'] == i and result['is_inside']:
                    is_inside = True
                    break
            
            # 选择线条样式：内部的用实线，外部的用虚线
            if is_inside:
                cv2.polylines(debug_img, [points], True, colors[i], 2)
            else:
                # 绘制虚线效果（通过多个小线段）
                for j in range(len(points)):
                    pt1 = points[j]
                    pt2 = points[(j + 1) % len(points)]
                    # 简单的虚线效果
                    mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.line(debug_img, tuple(pt1), mid_pt, colors[i], 2)
            
            # 在正方形中心添加编号和状态
            center_x = int(sum(x for x, y in square) / 4)
            center_y = int(sum(y for x, y in square) / 4)
            status = "IN" if is_inside else "OUT"
            cv2.putText(debug_img, f"S{i+1}({status})", 
                       (center_x-20, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
        
        # 添加信息
        info_color = (0, 255, 0) if stage == 'accepted_combinations' else (0, 0, 255)
        cv2.putText(debug_img, reason, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
        cv2.putText(debug_img, f"Line: {p1} -> {p2}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, info_color, 1)
        
        # 保存图像
        counter = self.debug_counters[stage]
        filename = os.path.join(self.debug_dirs[stage], f"{stage}_{counter:04d}.png")
        cv2.imwrite(filename, debug_img)
        print(f"保存调试图像: {filename}")
        
        # 递增计数器
        self.debug_counters[stage] += 1

    def _save_detected_vertices(self):
        """
        保存检测到的所有多边形顶点调试图像
        """
        if self.debug_frame is None:
            return
        
        # 创建调试图像
        debug_img = self.debug_frame.copy()
        
        # 绘制外轮廓（绿色）并标记顶点
        for contour_id, poly in enumerate(self.outer_polygons):
            vertices = poly['approx']
            # 绘制轮廓
            cv2.polylines(debug_img, [vertices], True, (0, 255, 0), 2)
            
            # 标记每个顶点
            for vertex_id, vertex in enumerate(vertices):
                coord = tuple(vertex[0])
                # 绘制顶点（绿色圆点）
                cv2.circle(debug_img, coord, 5, (0, 255, 0), -1)
                # 添加顶点编号
                cv2.putText(debug_img, f"O{contour_id}V{vertex_id}", 
                           (coord[0]+8, coord[1]-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 绘制内轮廓（红色）并标记顶点
        for contour_id, poly in enumerate(self.inner_polygons):
            vertices = poly['approx']
            # 绘制轮廓
            cv2.polylines(debug_img, [vertices], True, (0, 0, 255), 2)
            
            # 标记每个顶点
            for vertex_id, vertex in enumerate(vertices):
                coord = tuple(vertex[0])
                # 绘制顶点（红色圆点）
                cv2.circle(debug_img, coord, 5, (0, 0, 255), -1)
                # 添加顶点编号
                cv2.putText(debug_img, f"I{contour_id}V{vertex_id}", 
                           (coord[0]+8, coord[1]-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # 添加统计信息
        outer_vertex_count = sum(len(poly['approx']) for poly in self.outer_polygons)
        inner_vertex_count = sum(len(poly['approx']) for poly in self.inner_polygons)
        
        cv2.putText(debug_img, f"Outer: {len(self.outer_polygons)}, Vertices: {outer_vertex_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(debug_img, f"Inner: {len(self.inner_polygons)}, Vertices: {inner_vertex_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(debug_img, f"Total Vertices: {outer_vertex_count + inner_vertex_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 保存图像
        counter = self.debug_counters['detected_vertices']
        filename = os.path.join(self.debug_dirs['detected_vertices'], f"detected_vertices_{counter:04d}.png")
        cv2.imwrite(filename, debug_img)
        print(f"保存检测到的顶点调试图像: {filename}")
        
        # 递增计数器
        self.debug_counters['detected_vertices'] += 1


def test_integrated_poly_detection():
    """测试集成的多边形检测和测量功能"""
    print("初始化测量系统...")
    
    try:
        system = MeasurementSystem("calib.yaml", 500)
        print("系统初始化成功")
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return
    
    # 初始化多边形检测器
    poly_detector = IntegratedPolygonSquareDetector()
    
    print("开始集成多边形检测测试...")
    print("按 'q' 退出")
    
    while True:
        try:
            # 捕获帧
            frame = system.capture_frame()
            
            # 显示原始摄像头画面
            cv2.imshow("Camera Feed - Original", frame)
            
            # 预裁剪
            cropped_frame, ok = system.preprocessor.pre_crop(frame)
            if not ok:
                print("预裁剪失败，无法检测闭合轮廓")
                cv2.imshow("Polygon Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # 预处理（去畸变和边缘检测，使用裁剪后的帧）
            edges = system.preprocessor.preprocess(cropped_frame)
            
            # 检测A4纸边框并获取角点
            ok, corners = system.border_detector.detect_border(edges, cropped_frame)
            if not ok:
                print("无法检测A4边框")
                cv2.imshow("Polygon Detection", cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # 基于A4边框进行后裁切
            post_cropped_frame, adjusted_corners = system.border_detector.post_crop(cropped_frame, corners, inset_pixels=5)
            
            # 使用PnP计算距离D
            D, _ = system.distance_calculator.calculate_D(corners, system.K)
            if D is None:
                print("PnP求解失败")
                cv2.imshow("Polygon Detection", post_cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # 使用多边形检测器检测多边形
            outer_polygons, inner_polygons = poly_detector.detect_polygons(post_cropped_frame)
            
            # 创建结果显示图像
            result_frame = post_cropped_frame.copy()
            
            # 绘制检测到的多边形
            for poly in outer_polygons:
                cv2.polylines(result_frame, [poly['approx']], True, (0, 255, 0), 2)
            
            for poly in inner_polygons:
                cv2.polylines(result_frame, [poly['approx']], True, (0, 0, 255), 2)
            
            # 获取凸90度角顶点并绘制标记
            convex_90_vertices = poly_detector.get_90_degree_vertices(angle_tolerance=20)
            for vertex in convex_90_vertices:
                coord = vertex['coordinates']
                angle = vertex['angle']
                # 绘制凸90度角顶点为橙色圆点
                cv2.circle(result_frame, (int(coord[0]), int(coord[1])), 6, (0, 165, 255), -1)
                # 显示角度信息
                cv2.putText(result_frame, f"{angle:.1f}", 
                           (int(coord[0])+10, int(coord[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            # 查找最小边长的两点组合
            best_p1, best_p2, min_side_length_pix, all_combinations = poly_detector.find_minimum_square_points()
            
            if best_p1 is not None and best_p2 is not None:
                # 计算实际尺寸 - 使用类似shape_detector的方法
                # 这里我们使用像素边长来计算实际尺寸
                real_size = system.shape_detector.calculate_X(min_side_length_pix, D, system.K, adjusted_corners)
                
                print(f"检测到最小正方形: 像素边长: {min_side_length_pix:.2f}, 距离D: {D:.2f}cm, 实际边长: {real_size:.2f}cm")
                
                # 绘制最优线段
                cv2.line(result_frame, 
                        (int(best_p1[0]), int(best_p1[1])), 
                        (int(best_p2[0]), int(best_p2[1])), 
                        (255, 0, 255), 3)  # 紫色粗线
                
                # 绘制最优线段的端点
                cv2.circle(result_frame, (int(best_p1[0]), int(best_p1[1])), 8, (255, 0, 255), -1)
                cv2.circle(result_frame, (int(best_p2[0]), int(best_p2[1])), 8, (255, 0, 255), -1)
                
                # 构造并绘制最优正方形
                squares = poly_detector.create_squares_from_segment(best_p1, best_p2)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红、绿、蓝
                
                for i, square in enumerate(squares):
                    # 检查这个正方形是否在外轮廓内部
                    for poly in outer_polygons:
                        is_inside, _ = poly_detector.square_in_polygon(square, poly['approx'], tolerance=5.0)
                        if is_inside:
                            # 绘制在内部的正方形
                            points = np.array([(int(x), int(y)) for x, y in square], np.int32)
                            cv2.polylines(result_frame, [points], True, colors[i], 2)
                            break
                
                # 在图像上绘制测量信息
                cv2.putText(result_frame, f"{real_size:.2f}cm", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(result_frame, f"{D:.2f}cm", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            else:
                print("未检测到有效的凸90度角顶点或正方形")
                cv2.putText(result_frame, "No valid convex 90° vertices detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(result_frame, f"Convex 90° Vertices: {len(convex_90_vertices)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            
            # 显示结果
            cv2.imshow("Polygon Detection", result_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            continue
    
    # 清理资源
    cv2.destroyAllWindows()
    system.cap.release()
    print("测试结束")


if __name__ == "__main__":
    test_integrated_poly_detection()