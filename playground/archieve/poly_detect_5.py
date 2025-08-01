import cv2
import numpy as np
import math
from itertools import combinations


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
                 inner_epsilon_factor=0.02,
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
    
    def merge_close_vertices(self, vertices, distance_threshold=5.0):
        """合并位置差异太小的顶点"""
        if len(vertices) <= 2:
            return vertices
        
        points = np.array([vertex[0] for vertex in vertices], dtype=np.float32)
        n_points = len(points)
        merged = np.zeros(n_points, dtype=bool)
        merged_points = []
        
        for i in range(n_points):
            if merged[i]:
                continue
                
            current_point = points[i]
            close_indices = [i]
            
            for j in range(i + 1, n_points):
                if merged[j]:
                    continue
                    
                distance = np.linalg.norm(points[j] - current_point)
                if distance < distance_threshold:
                    close_indices.append(j)
            
            if len(close_indices) > 1:
                close_points = points[close_indices]
                merged_point = np.mean(close_points, axis=0)
                merged_points.append(merged_point)
                for idx in close_indices:
                    merged[idx] = True
            else:
                merged_points.append(current_point)
                merged[i] = True
        
        result = np.array([[[int(point[0]), int(point[1])]] for point in merged_points])
        return result
    
    def is_similar_contour(self, poly1, poly2):
        """判断两个轮廓是否相似（可能是同一形状的内外边缘）"""
        perimeter_ratio = abs(poly1['perimeter'] - poly2['perimeter']) / max(poly1['perimeter'], poly2['perimeter'])
        vertex_diff = abs(poly1['vertices'] - poly2['vertices'])
        return perimeter_ratio < self.similarity_perimeter_threshold and vertex_diff <= self.similarity_vertex_threshold
    
    def process_contours(self, contours, hierarchy):
        """处理轮廓：分类、拟合、过滤重复轮廓"""
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
                area = cv2.contourArea(contour)
                
                polygon_info = {
                    'contour': contour,
                    'approx': approx,
                    'vertices': len(approx),
                    'perimeter': perimeter,
                    'area': area,
                    'id': i,
                    'parent': parent,
                    'has_children': first_child != -1,
                    'is_outer': is_outer
                }
                
                all_polygons.append(polygon_info)
            
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
    
    def merge_polygon_vertices(self, outer_polygons, inner_polygons):
        """对所有多边形进行顶点合并处理"""
        for poly in outer_polygons:
            original_vertices = poly['approx']
            merged_vertices = self.merge_close_vertices(original_vertices, self.outer_vertex_merge_threshold)
            poly['approx'] = merged_vertices
            poly['vertices'] = len(merged_vertices)
        
        for poly in inner_polygons:
            original_vertices = poly['approx']
            merged_vertices = self.merge_close_vertices(original_vertices, self.inner_vertex_merge_threshold)
            poly['approx'] = merged_vertices
            poly['vertices'] = len(merged_vertices)
    
    def detect_polygons(self, frame):
        """主要的多边形检测方法"""
        self.image_shape = frame.shape
        edges = self.preprocess_image(frame)
        contours, hierarchy = self.find_contours_with_hierarchy(edges)
        all_polygons, outer_polygons, inner_polygons = self.process_contours(contours, hierarchy)
        self.merge_polygon_vertices(outer_polygons, inner_polygons)
        
        self.outer_polygons = outer_polygons
        self.inner_polygons = inner_polygons
        
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

    # ============ 正方形构造相关方法 ============
    
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
    
    def get_all_vertices(self):
        """获取所有外轮廓的顶点"""
        all_vertices = []
        for poly in self.outer_polygons:
            vertices = poly['approx']
            for vertex in vertices:
                coord = tuple(vertex[0])
                all_vertices.append(coord)
        return all_vertices
    
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
    
    def get_90_degree_vertices(self, angle_tolerance=10.0):
        """
        获取外轮廓中所有接近90度角的顶点
        
        Args:
            angle_tolerance: 角度容忍度（度数），默认10度
            
        Returns:
            list: 有效顶点列表
        """
        valid_vertices = []
        
        for contour_id, poly in enumerate(self.outer_polygons):
            vertices = poly['approx']
            n_vertices = len(vertices)
            
            for vertex_id in range(n_vertices):
                angle = self.get_vertex_angle('outer', contour_id, vertex_id)
                
                if angle is not None:
                    angle_diff = abs(angle - 90.0)
                    coordinates = tuple(vertices[vertex_id][0])
                    
                    if angle_diff <= angle_tolerance:
                        valid_vertices.append({
                            'contour_id': contour_id,
                            'vertex_id': vertex_id,
                            'angle': angle,
                            'coordinates': coordinates
                        })
        
        return valid_vertices

    def find_minimum_square_points(self):
        """
        找到构造正方形边长最小的两个点
        使用筛选后的90度角顶点进行组合：
        1. 获取所有90度角顶点
        2. 找到满足is_line_inside_shape的所有两点组合
        3. 用square_in_polygon确认是否有在外轮廓内部的构造正方形
        4. 如果有，求出边长
        5. 找到边长最小的那两个点
        """
        print("开始查找构造正方形边长最小的两个点...")
        
        # 步骤1：获取所有90度角顶点
        valid_vertices = self.get_90_degree_vertices(angle_tolerance=10.0)
        print(f"总共找到 {len(valid_vertices)} 个90度角顶点")
        
        if len(valid_vertices) < 2:
            print("90度角顶点数量不足，无法构造正方形")
            return None, None, float('inf'), []
        
        # 提取坐标用于组合
        vertex_coordinates = [vertex['coordinates'] for vertex in valid_vertices]
        
        # 步骤2：遍历所有两点组合
        valid_combinations = []
        total_combinations = len(list(combinations(vertex_coordinates, 2)))
        print(f"需要检查 {total_combinations} 个90度角顶点组合")
        
        processed = 0
        for p1, p2 in combinations(vertex_coordinates, 2):
            processed += 1
            if processed % 10 == 0:
                print(f"已处理 {processed}/{total_combinations} 个组合...")
            
            # 步骤3：检查连线是否在图形内部
            if not self.is_line_inside_shape(p1, p2):
                continue
            
            # 步骤4：构造三个正方形
            squares = self.create_squares_from_segment(p1, p2)
            
            # 步骤5：检查是否有正方形在外轮廓内部
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
            
            # 步骤6：如果有有效正方形，计算边长
            if has_valid_square:
                side_length = self.calculate_square_side_length(p1, p2)
                valid_combinations.append({
                    'point1': p1,
                    'point2': p2,
                    'side_length': side_length,
                    'squares': squares,
                    'square_results': square_results
                })
        
        print(f"找到 {len(valid_combinations)} 个有效的90度角顶点组合")
        
        if not valid_combinations:
            print("没有找到任何有效的90度角顶点组合")
            return None, None, float('inf'), []
        
        # 步骤7：找到边长最小的组合
        min_combination = min(valid_combinations, key=lambda x: x['side_length'])
        
        print(f"最小边长: {min_combination['side_length']:.2f}")
        print(f"最优两点: {min_combination['point1']} -> {min_combination['point2']}")
        
        return (min_combination['point1'], 
                min_combination['point2'], 
                min_combination['side_length'], 
                valid_combinations)
    
    def visualize_result(self, image, best_p1, best_p2, side_length, all_combinations):
        """可视化结果"""
        if best_p1 is None or best_p2 is None:
            print("没有有效结果需要可视化")
            return image
        
        result_image = image.copy()
        
        # 绘制所有外轮廓
        for poly in self.outer_polygons:
            cv2.polylines(result_image, [poly['approx']], True, (0, 255, 0), 2)
        
        # 绘制所有内轮廓
        for poly in self.inner_polygons:
            cv2.polylines(result_image, [poly['approx']], True, (0, 0, 255), 2)
        
        # 绘制最优线段
        cv2.line(result_image, 
                (int(best_p1[0]), int(best_p1[1])), 
                (int(best_p2[0]), int(best_p2[1])), 
                (255, 0, 255), 3)  # 紫色粗线
        
        # 绘制最优线段的端点
        cv2.circle(result_image, (int(best_p1[0]), int(best_p1[1])), 8, (255, 0, 255), -1)
        cv2.circle(result_image, (int(best_p2[0]), int(best_p2[1])), 8, (255, 0, 255), -1)
        
        # 构造并绘制最优正方形
        squares = self.create_squares_from_segment(best_p1, best_p2)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红、绿、蓝
        
        for i, square in enumerate(squares):
            # 检查这个正方形是否在外轮廓内部
            for poly in self.outer_polygons:
                is_inside, _ = self.square_in_polygon(square, poly['approx'], tolerance=5.0)
                if is_inside:
                    # 绘制在内部的正方形
                    points = np.array([(int(x), int(y)) for x, y in square], np.int32)
                    cv2.polylines(result_image, [points], True, colors[i], 2)
                    break
        
        # 添加文字信息
        cv2.putText(result_image, f"Min Side Length: {side_length:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(result_image, f"Valid Combinations: {len(all_combinations)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(result_image, f"Best Points: ({int(best_p1[0])},{int(best_p1[1])}) - ({int(best_p2[0])},{int(best_p2[1])})", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return result_image


def main():
    """主函数"""
    # 创建检测器
    detector = IntegratedPolygonSquareDetector()
    
    # 读取图像
    image_path = 'images/overlap/image3.png'
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        print(f"成功读取图像: {image.shape}")
        
        # 检测多边形
        print("正在检测多边形...")
        outer_polygons, inner_polygons = detector.detect_polygons(image)
        print(f"检测到 {len(outer_polygons)} 个外轮廓，{len(inner_polygons)} 个内轮廓")
        
        # 查找最小边长的两点组合
        best_p1, best_p2, min_side_length, all_combinations = detector.find_minimum_square_points()
        
        # 可视化结果
        result_image = detector.visualize_result(image, best_p1, best_p2, min_side_length, all_combinations)
        
        # 显示结果
        cv2.imshow('Original Image', image)
        cv2.imshow('Result', result_image)
        
        print("\n=== 最终结果 ===")
        if best_p1 is not None:
            print(f"最优两点: {best_p1} -> {best_p2}")
            print(f"最小边长: {min_side_length:.2f}")
            print(f"总共找到 {len(all_combinations)} 个有效组合")
        else:
            print("未找到满足条件的两点组合")
        
        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()