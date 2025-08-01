import cv2
import numpy as np
import math


class PolygonDetector:
    """
    多边形检测器类 - 检测图像中的内外轮廓并进行顶点优化
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
        初始化多边形检测器
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
        """
        主要的多边形检测方法
        
        Args:
            frame: 输入的BGR图像
            
        Returns:
            tuple: (outer_polygons, inner_polygons)
        """
        edges = self.preprocess_image(frame)
        contours, hierarchy = self.find_contours_with_hierarchy(edges)
        all_polygons, outer_polygons, inner_polygons = self.process_contours(contours, hierarchy)
        self.merge_polygon_vertices(outer_polygons, inner_polygons)
        
        self.outer_polygons = outer_polygons
        self.inner_polygons = inner_polygons
        
        return outer_polygons, inner_polygons
    
    def get_vertices(self):
        """
        获取所有检测到的顶点信息
        
        Returns:
            dict: 包含外轮廓和内轮廓顶点信息的字典
        """
        result = {
            'outer_vertices': [],
            'inner_vertices': []
        }
        
        for i, poly in enumerate(self.outer_polygons):
            vertices = [point[0].tolist() for point in poly['approx']]
            result['outer_vertices'].append({
                'id': i,
                'vertices': vertices,
                'vertex_count': len(vertices),
                'perimeter': poly['perimeter'],
                'area': poly['area']
            })
        
        for i, poly in enumerate(self.inner_polygons):
            vertices = [point[0].tolist() for point in poly['approx']]
            result['inner_vertices'].append({
                'id': i,
                'vertices': vertices,
                'vertex_count': len(vertices),
                'perimeter': poly['perimeter'],
                'area': poly['area'],
                'parent_id': poly['parent']
            })
        
        return result
    
    def _get_point_coordinate(self, point):
        """获取点的坐标，支持多种输入格式"""
        if isinstance(point, (tuple, list)) and len(point) == 2:
            try:
                return (float(point[0]), float(point[1]))
            except (ValueError, TypeError):
                return None
        
        if isinstance(point, dict):
            contour_type = point.get('contour_type')
            contour_id = point.get('contour_id', 0)
            vertex_id = point.get('vertex_id')
            
            if vertex_id is None:
                return None
            
            try:
                if contour_type == 'outer':
                    if contour_id < len(self.outer_polygons):
                        vertices = self.outer_polygons[contour_id]['approx']
                        if vertex_id < len(vertices):
                            coord = vertices[vertex_id][0]
                            return (float(coord[0]), float(coord[1]))
                
                elif contour_type == 'inner':
                    if contour_id < len(self.inner_polygons):
                        vertices = self.inner_polygons[contour_id]['approx']
                        if vertex_id < len(vertices):
                            coord = vertices[vertex_id][0]
                            return (float(coord[0]), float(coord[1]))
                
            except (IndexError, KeyError, TypeError, ValueError):
                return None
        
        return None

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
    
    def is_line_inside_shape(self, point1, point2):
        """
        判断两个点之间的连线是否完全在图形内部
        
        Args:
            point1: 起点坐标 (x, y) 或顶点序号
            point2: 终点坐标 (x, y) 或顶点序号
        
        Returns:
            bool: 连线是否在图形内部
        """
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


# 使用示例
def example_usage():
    """使用示例"""
    import cv2
    
    # 创建检测器
    detector = PolygonDetector()
    
    # 读取图像
    image = cv2.imread('images/overlap/image3.png')
    
    # 检测多边形
    outer_polygons, inner_polygons = detector.detect_polygons(image)
    
    # 获取顶点信息
    vertices_info = detector.get_vertices()
    
    # 获取90度角顶点
    right_angle_vertices = detector.get_90_degree_vertices(angle_tolerance=10.0)
    
    # 检测连线是否在图形内部
    is_inside = detector.is_line_inside_shape((100, 100), (200, 200))
    
    return detector, vertices_info, right_angle_vertices, is_inside


if __name__ == "__main__":
    example_usage()