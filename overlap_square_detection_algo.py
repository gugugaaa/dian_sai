import cv2
import numpy as np
import math
from itertools import combinations
from moving_average_filter import MovingAverageFilter

class OverlapSquareDetectionAlgorithm:
    """重叠正方形检测算法类 - 从多个重叠的正方形构成的多边形中,猜测并找到最小的正方形"""
    
    def __init__(self, measurement_system, **kwargs):
        """
        初始化重叠正方形检测算法
        
        Args:
            measurement_system: 测量系统实例
            **kwargs: 可调参数
                - filter_window_size: 移动平均值窗口大小 (默认: 10)
                - enable_size_filtering: 是否启用尺寸过滤 (默认: True)
                - min_square_size_cm: 最小正方形尺寸(cm) (默认: 1.0)
                - max_square_size_cm: 最大正方形尺寸(cm) (默认: 20.0)
                
                # 图像预处理参数
                - gaussian_blur_kernel: 高斯模糊核大小 (默认: (5, 5))
                - canny_low_threshold: Canny边缘检测低阈值 (默认: 50)
                - canny_high_threshold: Canny边缘检测高阈值 (默认: 150)
                
                # 轮廓检测参数
                - min_perimeter: 最小轮廓周长 (默认: 50)
                - outer_epsilon_factor: 外轮廓近似因子 (默认: 0.005)
                - inner_epsilon_factor: 内轮廓近似因子 (默认: 0.005)
                
                # 顶点合并参数
                - outer_vertex_merge_threshold: 外轮廓顶点合并阈值 (默认: 8.0)
                - inner_vertex_merge_threshold: 内轮廓顶点合并阈值 (默认: 6.0)
                
                # 相似性判断参数
                - similarity_perimeter_threshold: 周长相似性阈值 (默认: 0.1)
                - similarity_vertex_threshold: 顶点数相似性阈值 (默认: 2)
                
                # 角度检测参数
                - angle_tolerance: 90度角容忍度 (默认: 20)
                - convex_step_distance: 凸性检测步长 (默认: 10.0)
                - geometric_tolerance: 几何计算容忍度 (默认: 3.0)
        """
        
        # 存储测量系统引用
        self.measurement_system = measurement_system
        self.shape_detector = measurement_system.shape_detector
        
        # 可调参数
        self.config = {
            'filter_window_size': kwargs.get('filter_window_size', 10),
            'enable_size_filtering': kwargs.get('enable_size_filtering', True),
            'min_square_size_cm': kwargs.get('min_square_size_cm', 1.0),
            'max_square_size_cm': kwargs.get('max_square_size_cm', 20.0),
            
            # 图像预处理参数
            'gaussian_blur_kernel': kwargs.get('gaussian_blur_kernel', (5, 5)),
            'canny_low_threshold': kwargs.get('canny_low_threshold', 50),
            'canny_high_threshold': kwargs.get('canny_high_threshold', 150),
            
            # 轮廓检测参数
            'min_perimeter': kwargs.get('min_perimeter', 50),
            'outer_epsilon_factor': kwargs.get('outer_epsilon_factor', 0.005),
            'inner_epsilon_factor': kwargs.get('inner_epsilon_factor', 0.005),
            
            # 顶点合并参数
            'outer_vertex_merge_threshold': kwargs.get('outer_vertex_merge_threshold', 8.0),
            'inner_vertex_merge_threshold': kwargs.get('inner_vertex_merge_threshold', 6.0),
            
            # 相似性判断参数
            'similarity_perimeter_threshold': kwargs.get('similarity_perimeter_threshold', 0.1),
            'similarity_vertex_threshold': kwargs.get('similarity_vertex_threshold', 2),
            
            # 角度检测参数
            'angle_tolerance': kwargs.get('angle_tolerance', 20),
            'convex_step_distance': kwargs.get('convex_step_distance', 10.0),
            'geometric_tolerance': kwargs.get('geometric_tolerance', 3.0)
        }
        
        # 初始化组件和状态
        try:
            self.avg_filter = MovingAverageFilter(window_size=self.config['filter_window_size'])
            
            # 内部状态
            self.outer_polygons = []
            self.inner_polygons = []
            self.image_shape = None
            
            print("重叠正方形检测算法初始化成功")
            print(f"配置参数: {self.config}")
            
        except Exception as e:
            raise RuntimeError(f"重叠正方形检测算法初始化失败: {e}")
    
    def update_config(self, **kwargs):
        """动态更新配置参数"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"已更新参数 {key}: {value}")
            else:
                print(f"警告: 未知参数 {key}")
        
        # 如果更新了滤波器窗口大小，需要重新创建滤波器
        if 'filter_window_size' in kwargs:
            self.avg_filter = MovingAverageFilter(window_size=self.config['filter_window_size'])
    
    def process(self, post_cropped_frame, adjusted_corners, D_corrected, K):
        """
        处理post_crop后的图像，检测重叠/旋转正方形
        
        Args:
            post_cropped_frame: post_crop后的图像
            adjusted_corners: 调整后的A4纸角点
            D_corrected: 修正后的距离
            K: 相机内参矩阵
            
        Returns:
            dict: 包含检测结果的字典
        """
        try:
            # 1. 检测多边形
            self.image_shape = post_cropped_frame.shape
            self._detect_polygons(post_cropped_frame)
            
            if not self.outer_polygons:
                return {
                    'squares': [],
                    'polygons': {
                        'outer_count': 0,
                        'inner_count': 0
                    },
                    'statistics': {
                        'size_filtered': None,
                        'distance_filtered': D_corrected,
                        'filter_samples': len(self.avg_filter.size_history),
                        'convex_90_vertices': 0
                    },
                    'success': True,
                    'message': 'no valid outer polygons detected'
                }
            
            # 2. 获取凸90度角顶点
            convex_90_vertices = self._get_90_degree_vertices()
            
            if len(convex_90_vertices) < 2:
                return {
                    'squares': [],
                    'polygons': {
                        'outer_count': len(self.outer_polygons),
                        'inner_count': len(self.inner_polygons)
                    },
                    'statistics': {
                        'size_filtered': None,
                        'distance_filtered': D_corrected,
                        'filter_samples': len(self.avg_filter.size_history),
                        'convex_90_vertices': len(convex_90_vertices)
                    },
                    'success': True,
                    'message': f'detected {len(self.outer_polygons)} outer polygons, but <2 convex 90-degree vertices'
                }
            
            # 3. 找到最小正方形的两点组合
            best_p1, best_p2, min_side_length_pix, all_combinations = self._find_minimum_square_points()
            
            if best_p1 is None or best_p2 is None:
                return {
                    'squares': [],
                    'polygons': {
                        'outer_count': len(self.outer_polygons),
                        'inner_count': len(self.inner_polygons)
                    },
                    'statistics': {
                        'size_filtered': None,
                        'distance_filtered': D_corrected,
                        'filter_samples': len(self.avg_filter.size_history),
                        'convex_90_vertices': len(convex_90_vertices)
                    },
                    'success': True,
                    'message': f'detected {len(convex_90_vertices)} convex 90-degree vertices, but no valid square'
                }
            
            # 4. 计算实际尺寸
            real_size_raw = self.shape_detector.calculate_X(min_side_length_pix, D_corrected, K, adjusted_corners)
            
            # 5. 应用尺寸过滤
            filtered_out = False
            filter_reason = None
            
            if self.config['enable_size_filtering']:
                if (real_size_raw < self.config['min_square_size_cm'] or 
                    real_size_raw > self.config['max_square_size_cm']):
                    filtered_out = True
                    filter_reason = f"尺寸超出范围 [{self.config['min_square_size_cm']}, {self.config['max_square_size_cm']}]cm"
            
            # 6. 应用移动平均值滤波器
            real_size_filtered, D_filtered = None, D_corrected
            if not filtered_out:
                real_size_filtered, D_filtered = self.avg_filter.update(real_size_raw, D_corrected)
            
            # 7. 构造正方形
            squares = self._create_squares_from_segment(best_p1, best_p2)
            valid_squares = []
            
            for i, square in enumerate(squares):
                # 检查正方形是否在外轮廓内部
                for poly in self.outer_polygons:
                    is_inside, corner_results = self._square_in_polygon(square, poly['approx'])
                    if is_inside:
                        valid_squares.append({
                            'id': len(valid_squares),
                            'type': f'constructed_{i}',  # constructed_0, constructed_1, constructed_2
                            'corners': np.array([(int(x), int(y)) for x, y in square]),
                            'center': [int(sum(x for x, y in square) / 4), int(sum(y for x, y in square) / 4)],
                            'size_cm_raw': real_size_raw,
                            'size_cm': real_size_filtered if not filtered_out else real_size_raw,
                            'size_pixels': min_side_length_pix,
                            'construction_points': [best_p1, best_p2],
                            'detection_success': True,
                            'filtered_out': filtered_out,
                            'filter_reason': filter_reason,
                            'is_inside_polygon': True,
                            'polygon_id': poly['id']
                        })
                        break
            
            # 8. 构建返回结果
            result = {
                'squares': valid_squares,
                'polygons': {
                    'outer_count': len(self.outer_polygons),
                    'inner_count': len(self.inner_polygons),
                    'outer_polygons': self.outer_polygons,
                    'inner_polygons': self.inner_polygons
                },
                'construction_info': {
                    'best_points': [best_p1, best_p2] if best_p1 and best_p2 else None,
                    'total_combinations': len(all_combinations),
                    'convex_90_vertices': convex_90_vertices
                },
                'statistics': {
                    'size_filtered': real_size_filtered,
                    'distance_filtered': D_filtered,
                    'filter_samples': len(self.avg_filter.size_history),
                    'convex_90_vertices': len(convex_90_vertices)
                },
                'success': True,
                'message': f'detected {len(valid_squares)} squares, based on {len(convex_90_vertices)} convex 90-degree vertices'
            }
            
            return result
            
        except Exception as e:
            return {
                'squares': [],
                'polygons': {
                    'outer_count': 0,
                    'inner_count': 0
                },
                'statistics': {
                    'size_filtered': None,
                    'distance_filtered': D_corrected,
                    'filter_samples': len(self.avg_filter.size_history),
                    'convex_90_vertices': 0
                },
                'success': False,
                'message': f'error: {str(e)}'
            }
    
    def _detect_polygons(self, frame):
        """检测多边形"""
        edges = self._preprocess_image(frame)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_polygons, outer_polygons, inner_polygons = self._process_contours(contours, hierarchy)
        self.outer_polygons = outer_polygons
        self.inner_polygons = inner_polygons
    
    def _preprocess_image(self, frame):
        """图像预处理：灰度化 -> 高斯模糊 -> OTSU阈值 -> Canny边缘检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config['gaussian_blur_kernel'], 0)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(otsu_thresh, self.config['canny_low_threshold'], self.config['canny_high_threshold'])
        return edges
    
    def _process_contours(self, contours, hierarchy):
        """处理轮廓：分类、拟合、过滤重复轮廓、合并邻近顶点"""
        all_polygons = []
        outer_polygons = []
        inner_polygons = []
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            
            for i, contour in enumerate(contours):
                perimeter = cv2.arcLength(contour, True)
                if perimeter < self.config['min_perimeter']:
                    continue
                
                next_contour, prev_contour, first_child, parent = hierarchy[i]
                is_outer = parent == -1
                
                epsilon = (self.config['outer_epsilon_factor'] if is_outer else self.config['inner_epsilon_factor']) * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 顶点合并
                merge_threshold = (self.config['outer_vertex_merge_threshold'] if is_outer else 
                                 self.config['inner_vertex_merge_threshold'])
                merged_approx = self._merge_polygon_vertices(approx, merge_threshold)
                area = cv2.contourArea(merged_approx)
                
                polygon_info = {
                    'contour': contour,
                    'approx': merged_approx,
                    'vertices': len(merged_approx),
                    'perimeter': perimeter,
                    'area': area,
                    'id': i,
                    'parent': parent,
                    'has_children': first_child != -1,
                    'is_outer': is_outer
                }
                
                all_polygons.append(polygon_info)
            
            # 分类轮廓
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
                    
                    if parent_poly and self._is_similar_contour(poly, parent_poly):
                        continue
                    else:
                        inner_polygons.append(poly)
        
        return all_polygons, outer_polygons, inner_polygons
    
    def _merge_polygon_vertices(self, vertices, distance_threshold=5.0):
        """合并位置差异太小的顶点，保持原有轮廓的顺序"""
        if len(vertices) <= 2:
            return vertices
        
        points = np.array([vertex[0] for vertex in vertices], dtype=np.float32)
        n_points = len(points)
        
        if n_points == 0:
            return vertices
        
        to_merge = []
        merged_indices = set()
        
        i = 0
        while i < n_points:
            if i in merged_indices:
                i += 1
                continue
                
            current_point = points[i]
            merge_group = [i]
            
            j = (i + 1) % n_points
            consecutive_count = 0
            
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
            
            if len(merge_group) > 1:
                j = (i - 1) % n_points
                consecutive_count = 0
                
                while j != i and consecutive_count < n_points - 1 and j not in merged_indices:
                    distance = np.linalg.norm(points[j] - current_point)
                    if distance < distance_threshold:
                        merge_group.insert(0, j)
                        merged_indices.add(j)
                        j = (j - 1) % n_points
                        consecutive_count += 1
                    else:
                        break
            
            to_merge.append(merge_group)
            merged_indices.add(i)
            i += 1
        
        merged_vertices = []
        processed_indices = set()
        
        for i in range(n_points):
            if i in processed_indices:
                continue
                
            merge_group = None
            for group in to_merge:
                if i in group:
                    merge_group = group
                    break
            
            if merge_group and len(merge_group) > 1:
                group_points = points[merge_group]
                merged_point = np.mean(group_points, axis=0)
                merged_vertices.append(merged_point)
                
                for idx in merge_group:
                    processed_indices.add(idx)
            else:
                merged_vertices.append(points[i])
                processed_indices.add(i)
        
        if len(merged_vertices) == 0:
            return vertices
            
        result = np.array([[[int(point[0]), int(point[1])]] for point in merged_vertices])
        return result
    
    def _is_similar_contour(self, poly1, poly2):
        """判断两个轮廓是否相似"""
        perimeter_ratio = abs(poly1['perimeter'] - poly2['perimeter']) / max(poly1['perimeter'], poly2['perimeter'])
        vertex_diff = abs(poly1['vertices'] - poly2['vertices'])
        return (perimeter_ratio < self.config['similarity_perimeter_threshold'] and 
                vertex_diff <= self.config['similarity_vertex_threshold'])
    
    def _get_90_degree_vertices(self):
        """获取外轮廓中所有接近90度角的凸顶点"""
        valid_vertices = []
        
        for contour_id, poly in enumerate(self.outer_polygons):
            vertices = poly['approx']
            n_vertices = len(vertices)
            
            for vertex_id in range(n_vertices):
                coordinates = tuple(vertices[vertex_id][0])
                
                # 检查是否是凸点
                is_convex = self._is_convex_vertex('outer', contour_id, vertex_id)
                if not is_convex:
                    continue
                
                # 检查角度
                angle = self._get_vertex_angle('outer', contour_id, vertex_id)
                
                if angle is not None:
                    angle_diff = abs(angle - 90.0)
                    
                    if angle_diff <= self.config['angle_tolerance']:
                        valid_vertices.append({
                            'contour_id': contour_id,
                            'vertex_id': vertex_id,
                            'angle': angle,
                            'coordinates': coordinates,
                            'is_convex': True
                        })
        
        return valid_vertices
    
    def _is_convex_vertex(self, contour_type, contour_id, vertex_id):
        """检测指定顶点是否是凸点"""
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
            
            vector1 = prev_point - curr_point
            vector2 = next_point - curr_point
            
            len1 = np.linalg.norm(vector1)
            len2 = np.linalg.norm(vector2)
            
            if len1 == 0 or len2 == 0:
                return False
            
            vector1_norm = vector1 / len1
            vector2_norm = vector2 / len2
            
            bisector = vector1_norm + vector2_norm
            bisector_len = np.linalg.norm(bisector)
            
            if bisector_len == 0:
                return False
            
            bisector_norm = bisector / bisector_len
            test_point = curr_point + bisector_norm * self.config['convex_step_distance']
            
            return self._point_in_valid_area_with_tolerance(tuple(test_point))
            
        except Exception:
            return False
    
    def _get_vertex_angle(self, contour_type, contour_id, vertex_id):
        """计算指定顶点的两条连接边的夹角"""
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
    
    def _find_minimum_square_points(self):
        """找到构造正方形边长最小的两个点"""
        valid_vertices = self._get_90_degree_vertices()
        
        if len(valid_vertices) < 2:
            return None, None, float('inf'), []
        
        vertex_coordinates = [vertex['coordinates'] for vertex in valid_vertices]
        valid_combinations = []
        
        for p1, p2 in combinations(vertex_coordinates, 2):
            if not self._is_line_inside_shape(p1, p2):
                continue
            
            squares = self._create_squares_from_segment(p1, p2)
            has_valid_square = False
            square_results = []
            
            for i, square in enumerate(squares):
                for poly in self.outer_polygons:
                    is_inside, corner_results = self._square_in_polygon(square, poly['approx'])
                    square_results.append({
                        'square_index': i,
                        'polygon_index': poly['id'],
                        'is_inside': is_inside,
                        'corner_results': corner_results
                    })
                    
                    if is_inside:
                        has_valid_square = True
            
            if has_valid_square:
                side_length = self._calculate_square_side_length(p1, p2)
                valid_combinations.append({
                    'point1': p1,
                    'point2': p2,
                    'side_length': side_length,
                    'squares': squares,
                    'square_results': square_results
                })
        
        if not valid_combinations:
            return None, None, float('inf'), []
        
        min_combination = min(valid_combinations, key=lambda x: x['side_length'])
        
        return (min_combination['point1'], 
                min_combination['point2'], 
                min_combination['side_length'], 
                valid_combinations)
    
    def _create_squares_from_segment(self, p1, p2):
        """根据线段端点创建三个正方形"""
        x1, y1 = p1
        x2, y2 = p2
        
        dx = x2 - x1
        dy = y2 - y1
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
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        half_diag = math.sqrt(dx*dx + dy*dy) / 2
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
    
    def _calculate_square_side_length(self, p1, p2):
        """计算由两点构成的正方形的边长"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _square_in_polygon(self, square, polygon_contour, tolerance=5.0):
        """检测正方形是否在多边形内部"""
        corner_results = []
        all_inside = True
        
        for i, corner in enumerate(square):
            is_inside = self._point_in_polygon_with_tolerance(corner, polygon_contour, tolerance)
            corner_results.append({
                'corner_index': i,
                'position': corner,
                'is_inside': is_inside
            })
            if not is_inside:
                all_inside = False
        
        return all_inside, corner_results
    
    def _point_in_polygon_with_tolerance(self, point, polygon_contour, tolerance=5.0):
        """检查点是否在多边形内部，带容忍度处理"""
        if isinstance(polygon_contour, list):
            contour = np.array([(int(x), int(y)) for x, y in polygon_contour], dtype=np.int32)
        else:
            contour = polygon_contour
        
        distance = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
        return distance >= -tolerance
    
    def _is_line_inside_shape(self, point1, point2):
        """判断两个点之间的连线是否完全在图形内部"""
        p1 = self._get_point_coordinate(point1)
        p2 = self._get_point_coordinate(point2)
        
        if p1 is None or p2 is None:
            return False
        
        tolerance = self.config['geometric_tolerance']
        
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
        
        # 检查线段上的采样点
        sample_points = [
            (p1[0] + 0.25 * (p2[0] - p1[0]), p1[1] + 0.25 * (p2[1] - p1[1])),
            (p1[0] + 0.5 * (p2[0] - p1[0]), p1[1] + 0.5 * (p2[1] - p1[1])),
            (p1[0] + 0.75 * (p2[0] - p1[0]), p1[1] + 0.75 * (p2[1] - p1[1]))
        ]
        
        for sample_point in sample_points:
            if not self._point_in_valid_area_with_tolerance(sample_point, tolerance):
                return False
        
        return True
    
    def _get_point_coordinate(self, point):
        """获取点的坐标，支持多种输入格式"""
        if isinstance(point, (tuple, list)) and len(point) == 2:
            try:
                return (float(point[0]), float(point[1]))
            except (ValueError, TypeError):
                return None
        return None
    
    def _point_in_valid_area_with_tolerance(self, point, tolerance=None):
        """检查点是否在有效区域内，带容忍度处理"""
        if tolerance is None:
            tolerance = self.config['geometric_tolerance']
            
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
    
    def _line_breaks_through_outer_contour(self, p1, p2, contour, tolerance):
        """判断线段是否冲破外轮廓"""
        p1_dist = cv2.pointPolygonTest(contour, p1, True)
        p2_dist = cv2.pointPolygonTest(contour, p2, True)
        
        p1_outside = p1_dist < -tolerance
        p2_outside = p2_dist < -tolerance
        
        if not (p1_outside or p2_outside):
            return False
        
        intersections = self._get_line_contour_intersections(p1, p2, contour)
        return len(intersections) > 0
    
    def _line_breaks_through_inner_contour(self, p1, p2, contour, tolerance):
        """判断线段是否冲破内轮廓"""
        intersections = self._get_line_contour_intersections(p1, p2, contour)
        
        if len(intersections) == 0:
            return False
        
        return self._line_crosses_inner_contour(p1, p2, contour, intersections, tolerance)
    
    def _line_crosses_inner_contour(self, p1, p2, contour, intersections, tolerance):
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
    
    def reset_filter(self):
        """重置移动平均值滤波器"""
        self.avg_filter.reset()
        print("已重置移动平均值滤波器")
    
    def get_config(self):
        """获取当前配置"""
        return self.config.copy()
    
    def get_statistics(self):
        """获取当前统计信息"""
        return {
            'filter_samples': len(self.avg_filter.size_history),
            'size_history': self.avg_filter.size_history.copy(),
            'distance_history': self.avg_filter.distance_history.copy(),
            'polygon_counts': {
                'outer': len(self.outer_polygons),
                'inner': len(self.inner_polygons)
            }
        }