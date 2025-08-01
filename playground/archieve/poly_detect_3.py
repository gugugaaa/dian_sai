import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import math

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
        
        Args:
            gaussian_blur_kernel: 高斯模糊核大小
            canny_low_threshold: Canny边缘检测低阈值
            canny_high_threshold: Canny边缘检测高阈值
            min_perimeter: 最小轮廓周长
            outer_epsilon_factor: 外轮廓多边形拟合精度系数
            inner_epsilon_factor: 内轮廓多边形拟合精度系数
            outer_vertex_merge_threshold: 外轮廓顶点合并距离阈值
            inner_vertex_merge_threshold: 内轮廓顶点合并距离阈值
            similarity_perimeter_threshold: 轮廓相似性周长阈值
            similarity_vertex_threshold: 轮廓相似性顶点数阈值
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
        
        # 存储处理结果和调试图像
        self.original_frame = None
        self.processed_edges = None
        self.debug_images = {}
        self.outer_polygons = []
        self.inner_polygons = []
        
        # 初始化中文字体
        self.font = self._init_chinese_font()
        self.small_font = self._init_chinese_font(size=12)
    
    def _init_chinese_font(self, size=16):
        """初始化中文字体"""
        font_paths = [
            "simhei.ttf",
            "C:/Windows/Fonts/simhei.ttf", 
            "C:/Windows/Fonts/msyh.ttc",
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        # 使用默认字体
        return ImageFont.load_default()
    
    def put_chinese_text(self, img, text, position, font=None, color=(255, 255, 255)):
        """
        在OpenCV图像上绘制中文文本
        
        Args:
            img: OpenCV图像 (BGR格式)
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font: PIL字体对象
            color: 文本颜色 (B, G, R)
        """
        if font is None:
            font = self.font
            
        # 转换颜色格式 BGR -> RGB
        rgb_color = (color[2], color[1], color[0])
        
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=rgb_color)
        
        # 转换回OpenCV格式
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 复制回原图像
        img[:] = img_cv[:]

    def preprocess_image(self, frame):
        """
        图像预处理：灰度化 -> 高斯模糊 -> OTSU阈值 -> Canny边缘检测
        
        Args:
            frame: 输入的BGR图像
            
        Returns:
            edges: Canny边缘检测结果
            gray: 灰度图
            otsu_thresh: OTSU阈值化结果
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_kernel, 0)
        
        # OTSU阈值处理
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 边缘检测
        edges = cv2.Canny(otsu_thresh, self.canny_low_threshold, self.canny_high_threshold)
        
        return edges, gray, otsu_thresh
    
    def find_contours_with_hierarchy(self, edges):
        """
        查找轮廓并获取层次结构信息
        
        Args:
            edges: Canny边缘检测结果
            
        Returns:
            contours: 检测到的轮廓列表
            hierarchy: 轮廓层次结构信息
        """
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    
    def merge_close_vertices(self, vertices, distance_threshold=5.0):
        """
        合并位置差异太小的顶点
        
        Args:
            vertices: 顶点列表，OpenCV格式 [[[x, y]], [[x, y]], ...]
            distance_threshold: 距离阈值，小于此距离的顶点将被合并
        
        Returns:
            merged_vertices: 合并后的顶点列表，保持OpenCV格式
        """
        if len(vertices) <= 2:
            return vertices
        
        # 将顶点转换为更容易处理的格式
        points = np.array([vertex[0] for vertex in vertices], dtype=np.float32)
        n_points = len(points)
        
        # 标记哪些点已经被合并
        merged = np.zeros(n_points, dtype=bool)
        merged_points = []
        
        for i in range(n_points):
            if merged[i]:
                continue
                
            # 找到与当前点距离小于阈值的所有点
            current_point = points[i]
            close_indices = [i]
            
            for j in range(i + 1, n_points):
                if merged[j]:
                    continue
                    
                distance = np.linalg.norm(points[j] - current_point)
                if distance < distance_threshold:
                    close_indices.append(j)
            
            # 如果找到了需要合并的点
            if len(close_indices) > 1:
                # 计算这些点的均值
                close_points = points[close_indices]
                merged_point = np.mean(close_points, axis=0)
                merged_points.append(merged_point)
                
                # 标记这些点为已合并
                for idx in close_indices:
                    merged[idx] = True
                    
                print(f"合并了 {len(close_indices)} 个顶点: {close_indices}, 合并后位置: ({merged_point[0]:.1f}, {merged_point[1]:.1f})")
            else:
                # 没有需要合并的点，直接添加
                merged_points.append(current_point)
                merged[i] = True
        
        # 转换回OpenCV格式
        result = np.array([[[int(point[0]), int(point[1])]] for point in merged_points])
        
        print(f"顶点合并: {len(vertices)} -> {len(result)} 个顶点")
        return result
    
    def is_similar_contour(self, poly1, poly2):
        """
        判断两个轮廓是否相似（可能是同一形状的内外边缘）
        
        Args:
            poly1, poly2: 多边形信息字典
            
        Returns:
            bool: 是否相似
        """
        # 周长相似度检查
        perimeter_ratio = abs(poly1['perimeter'] - poly2['perimeter']) / max(poly1['perimeter'], poly2['perimeter'])
        
        # 顶点数相似度检查
        vertex_diff = abs(poly1['vertices'] - poly2['vertices'])
        
        return perimeter_ratio < self.similarity_perimeter_threshold and vertex_diff <= self.similarity_vertex_threshold
    
    def process_contours(self, contours, hierarchy):
        """
        处理轮廓：分类、拟合、过滤重复轮廓
        
        Args:
            contours: OpenCV检测到的轮廓列表
            hierarchy: 轮廓层次结构信息
            
        Returns:
            all_polygons: 所有轮廓信息列表
            outer_polygons: 过滤后的外轮廓列表
            inner_polygons: 过滤后的内轮廓列表
        """
        all_polygons = []
        outer_polygons = []
        inner_polygons = []
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]  # hierarchy的形状是(1, n, 4)，取第一维
            
            # 第一步：收集所有轮廓信息
            for i, contour in enumerate(contours):
                # 计算轮廓周长
                perimeter = cv2.arcLength(contour, True)
                
                # 跳过过小的轮廓
                if perimeter < self.min_perimeter:
                    continue
                
                # 分析层次结构
                # hierarchy[i] = [next, previous, first_child, parent]
                next_contour, prev_contour, first_child, parent = hierarchy[i]
                
                # 判断是外轮廓还是内轮廓
                is_outer = parent == -1  # 没有父轮廓的是外轮廓
                is_inner = parent != -1  # 有父轮廓的是内轮廓
                
                # 根据内外轮廓使用不同的拟合精度
                if is_outer:
                    epsilon = self.outer_epsilon_factor * perimeter
                else:
                    epsilon = self.inner_epsilon_factor * perimeter
                
                # 多边形拟合
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                
                # 存储轮廓信息
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
            
            # 第二步：分类并过滤轮廓
            for poly in all_polygons:
                if poly['is_outer']:
                    outer_polygons.append(poly)
                else:
                    # 对于内轮廓，检查是否与父轮廓重复
                    parent_id = poly['parent']
                    parent_poly = None
                    
                    # 找到父轮廓
                    for p in all_polygons:
                        if p['id'] == parent_id:
                            parent_poly = p
                            break
                    
                    # 如果找到父轮廓，检查相似性
                    if parent_poly and self.is_similar_contour(poly, parent_poly):
                        # 跳过与父轮廓相似的内轮廓（重复轮廓）
                        print(f"过滤重复轮廓: 内轮廓{poly['id']} 与父轮廓{parent_id} 相似")
                        continue
                    else:
                        # 保留真正的内轮廓
                        inner_polygons.append(poly)
        
        return all_polygons, outer_polygons, inner_polygons
    
    def merge_polygon_vertices(self, outer_polygons, inner_polygons):
        """
        对所有多边形进行顶点合并处理
        
        Args:
            outer_polygons: 外轮廓列表
            inner_polygons: 内轮廓列表
        """
        print("\n=== 开始顶点合并处理 ===")
        
        # 处理外轮廓
        for i, poly in enumerate(outer_polygons):
            print(f"\n处理外轮廓 {i+1}:")
            original_vertices = poly['approx']
            merged_vertices = self.merge_close_vertices(original_vertices, self.outer_vertex_merge_threshold)
            poly['approx'] = merged_vertices
            poly['vertices'] = len(merged_vertices)
        
        # 处理内轮廓
        for i, poly in enumerate(inner_polygons):
            print(f"\n处理内轮廓 {i+1}:")
            original_vertices = poly['approx']
            merged_vertices = self.merge_close_vertices(original_vertices, self.inner_vertex_merge_threshold)
            poly['approx'] = merged_vertices
            poly['vertices'] = len(merged_vertices)
    
    def create_debug_images(self, frame, edges, all_polygons, outer_polygons, inner_polygons):
        """
        创建各种调试图像
        
        Args:
            frame: 原始图像
            edges: 边缘检测结果
            all_polygons: 所有轮廓信息
            outer_polygons: 外轮廓列表
            inner_polygons: 内轮廓列表
        """
        # 创建调试图像
        edge_debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        poly_debug = frame.copy()
        hierarchy_debug = frame.copy()
        filtered_debug = frame.copy()
        vertex_merge_debug = frame.copy()
        combined_display = frame.copy()
        
        # 绘制所有轮廓（包括被过滤的）
        for poly in all_polygons:
            color = (0, 255, 0) if poly['is_outer'] else (255, 0, 0)  # 外轮廓绿色，内轮廓红色
            cv2.drawContours(edge_debug, [poly['contour']], -1, color, 2)
            cv2.drawContours(poly_debug, [poly['approx']], -1, color, 2)
            
            # 计算轮廓中心用于标注
            M = cv2.moments(poly['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 标注信息 - 使用中文文本函数
                label = f"{'外' if poly['is_outer'] else '内'}:{poly['vertices']}"
                self.put_chinese_text(poly_debug, label, (cx-20, cy-20), self.small_font, color)
                
                detail_label = f"ID:{poly['id']} P:{poly['parent']}"
                self.put_chinese_text(hierarchy_debug, detail_label, (cx-30, cy-30), self.small_font, color)
            
            # 绘制顶点
            for point in poly['approx']:
                cv2.circle(poly_debug, tuple(point[0]), 3, color, -1)
        
        # 绘制过滤后的外轮廓
        for poly in outer_polygons:
            cv2.drawContours(filtered_debug, [poly['approx']], -1, (0, 255, 0), 3)
            cv2.drawContours(vertex_merge_debug, [poly['approx']], -1, (0, 255, 0), 3)
            cv2.drawContours(combined_display, [poly['approx']], -1, (0, 255, 0), 3)
            
            M = cv2.moments(poly['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.put_chinese_text(filtered_debug, f"外:{poly['vertices']}", (cx-20, cy-40), self.font, (0, 255, 0))
                self.put_chinese_text(vertex_merge_debug, f"外:{poly['vertices']}", (cx-20, cy-40), self.font, (0, 255, 0))
                self.put_chinese_text(combined_display, f"外{len(outer_polygons)}", (cx-15, cy-40), self.font, (0, 255, 0))
            
            # 绘制合并后的顶点（用较大的圆圈标识）
            for point in poly['approx']:
                cv2.circle(vertex_merge_debug, tuple(point[0]), 5, (0, 255, 0), -1)
                cv2.circle(vertex_merge_debug, tuple(point[0]), 7, (255, 255, 255), 2)
        
        # 绘制过滤后的内轮廓
        for i, poly in enumerate(inner_polygons):
            cv2.drawContours(filtered_debug, [poly['approx']], -1, (0, 0, 255), 2)
            cv2.drawContours(vertex_merge_debug, [poly['approx']], -1, (0, 0, 255), 2)
            cv2.drawContours(combined_display, [poly['approx']], -1, (0, 0, 255), 2)
            
            M = cv2.moments(poly['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.put_chinese_text(filtered_debug, f"内:{poly['vertices']}", (cx-20, cy+15), self.small_font, (0, 0, 255))
                self.put_chinese_text(vertex_merge_debug, f"内:{poly['vertices']}", (cx-20, cy+15), self.small_font, (0, 0, 255))
                self.put_chinese_text(combined_display, f"内{i+1}", (cx-15, cy+10), self.small_font, (0, 0, 255))
            
            # 绘制合并后的顶点
            for point in poly['approx']:
                cv2.circle(vertex_merge_debug, tuple(point[0]), 4, (0, 0, 255), -1)
                cv2.circle(vertex_merge_debug, tuple(point[0]), 6, (255, 255, 255), 2)
        
        # 存储调试图像
        self.debug_images = {
            'original': frame,
            'edges': edges,
            'edge_debug': edge_debug,
            'poly_debug': poly_debug,
            'hierarchy_debug': hierarchy_debug,
            'filtered_debug': filtered_debug,
            'vertex_merge_debug': vertex_merge_debug,
            'combined_display': combined_display
        }
    
    def detect_polygons(self, frame):
        """
        主要的多边形检测方法 - 完整的处理流程
        
        Args:
            frame: 输入的BGR图像
            
        Returns:
            outer_polygons: 检测到的外轮廓列表
            inner_polygons: 检测到的内轮廓列表
        """
        # 保存原始图像
        self.original_frame = frame.copy()
        
        # 1. 图像预处理
        edges, gray, otsu_thresh = self.preprocess_image(frame)
        self.processed_edges = edges
        
        # 2. 查找轮廓
        contours, hierarchy = self.find_contours_with_hierarchy(edges)
        
        # 3. 处理轮廓
        all_polygons, outer_polygons, inner_polygons = self.process_contours(contours, hierarchy)
        
        # 4. 顶点合并
        self.merge_polygon_vertices(outer_polygons, inner_polygons)
        
        # 5. 创建调试图像
        self.create_debug_images(frame, edges, all_polygons, outer_polygons, inner_polygons)
        
        # 6. 保存结果
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
        
        # 提取外轮廓顶点
        for i, poly in enumerate(self.outer_polygons):
            vertices = [point[0].tolist() for point in poly['approx']]
            result['outer_vertices'].append({
                'id': i,
                'vertices': vertices,
                'vertex_count': len(vertices),
                'perimeter': poly['perimeter'],
                'area': poly['area']
            })
        
        # 提取内轮廓顶点
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
        """
        获取点的坐标，支持多种输入格式
        
        Args:
            point: 可以是：
                   - 坐标元组: (x, y)
                   - 顶点描述字典: {'contour_type': 'outer'/'inner', 'contour_id': int, 'vertex_id': int}
                   
        Returns:
            tuple: (x, y) 坐标，如果无效则返回 None
        """
        # 如果已经是坐标元组
        if isinstance(point, (tuple, list)) and len(point) == 2:
            try:
                return (float(point[0]), float(point[1]))
            except (ValueError, TypeError):
                return None
        
        # 如果是顶点描述字典
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

    # ===== 角度计算相关方法 =====
    
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
            # 获取对应的轮廓
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
            
            # 获取三个点：前一个、当前、后一个
            prev_idx = (vertex_id - 1) % n_vertices
            curr_idx = vertex_id
            next_idx = (vertex_id + 1) % n_vertices
            
            prev_point = vertices[prev_idx][0].astype(float)
            curr_point = vertices[curr_idx][0].astype(float)
            next_point = vertices[next_idx][0].astype(float)
            
            # 计算两个向量
            vector1 = prev_point - curr_point  # 从当前顶点指向前一个顶点
            vector2 = next_point - curr_point  # 从当前顶点指向后一个顶点
            
            # 计算向量长度
            len1 = np.linalg.norm(vector1)
            len2 = np.linalg.norm(vector2)
            
            if len1 == 0 or len2 == 0:
                return None
            
            # 计算点积
            dot_product = np.dot(vector1, vector2)
            
            # 计算夹角（弧度）
            cos_angle = dot_product / (len1 * len2)
            
            # 防止浮点数精度问题导致的数值溢出
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # 转换为角度
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            print(f"计算顶点角度时发生错误: {e}")
            return None
    
    def get_90_degree_vertices(self, angle_tolerance=10.0):
        """
        获取外轮廓中所有接近90度角的顶点
        
        Args:
            angle_tolerance: 角度容忍度（度数），默认10度
            
        Returns:
            list: 有效顶点列表，每个元素为字典：
                  {'contour_id': int, 'vertex_id': int, 'angle': float, 'coordinates': (x, y)}
        """
        valid_vertices = []
        
        print(f"\n=== 检测90度角顶点（容忍度: ±{angle_tolerance}°）===")
        
        # 遍历所有外轮廓
        for contour_id, poly in enumerate(self.outer_polygons):
            vertices = poly['approx']
            n_vertices = len(vertices)
            
            print(f"\n检查外轮廓 {contour_id}（共 {n_vertices} 个顶点）:")
            
            # 遍历该轮廓的所有顶点
            for vertex_id in range(n_vertices):
                angle = self.get_vertex_angle('outer', contour_id, vertex_id)
                
                if angle is not None:
                    # 检查是否接近90度
                    angle_diff = abs(angle - 90.0)
                    
                    coordinates = tuple(vertices[vertex_id][0])
                    
                    print(f"  顶点 {vertex_id}: 角度 {angle:.1f}°, 坐标 {coordinates}, "
                          f"与90°差值: {angle_diff:.1f}°", end="")
                    
                    if angle_diff <= angle_tolerance:
                        valid_vertices.append({
                            'contour_id': contour_id,
                            'vertex_id': vertex_id,
                            'angle': angle,
                            'coordinates': coordinates
                        })
                        print(" ✓有效")
                    else:
                        print(" ✗无效")
                else:
                    print(f"  顶点 {vertex_id}: 计算角度失败")
        
        print(f"\n找到 {len(valid_vertices)} 个接近90度的顶点")
        return valid_vertices
    
    def print_detection_summary(self):
        """打印检测结果摘要"""
        print(f"\n=== 检测结果摘要 ===")
        print(f"检测到 {len(self.outer_polygons)} 个外轮廓:")
        for j, poly in enumerate(self.outer_polygons):
            print(f"  外轮廓 {j+1}: {poly['vertices']} 个顶点, 周长 {poly['perimeter']:.1f}, "
                  f"面积 {poly['area']:.1f}, 有子轮廓: {'是' if poly['has_children'] else '否'}")
        
        print(f"检测到 {len(self.inner_polygons)} 个内轮廓:")
        for j, poly in enumerate(self.inner_polygons):
            print(f"  内轮廓 {j+1}: {poly['vertices']} 个顶点, 周长 {poly['perimeter']:.1f}, "
                  f"面积 {poly['area']:.1f}, 父轮廓ID: {poly['parent']}")
    
    def visualize_results(self, image_name="多边形检测结果"):
        """
        可视化检测结果
        
        Args:
            image_name: 图像标题
        """
        if not self.debug_images:
            print("请先调用detect_polygons方法进行检测")
            return
        
        # 使用matplotlib显示所有图像（3x3布局）
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(image_name, fontsize=16)
        
        # 第一行
        axes[0, 0].imshow(cv2.cvtColor(self.debug_images['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原始图片')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.debug_images['edges'], cmap='gray')
        axes[0, 1].set_title('Canny边缘检测')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(self.debug_images['edge_debug'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('所有轮廓检测\n(绿色:外轮廓 红色:内轮廓)')
        axes[0, 2].axis('off')
        
        # 第二行
        axes[1, 0].imshow(cv2.cvtColor(self.debug_images['poly_debug'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('多边形拟合')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(self.debug_images['hierarchy_debug'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('层次结构')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(self.debug_images['filtered_debug'], cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('过滤后轮廓\n(去除重复轮廓)')
        axes[1, 2].axis('off')
        
        # 第三行
        axes[2, 0].imshow(cv2.cvtColor(self.debug_images['vertex_merge_debug'], cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title('顶点合并后\n(白圈标识合并顶点)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(cv2.cvtColor(self.debug_images['combined_display'], cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title('内外轮廓融合显示')
        axes[2, 1].axis('off')
        
        # 最后一个位置显示统计信息
        axes[2, 2].text(0.1, 0.9, f'外轮廓数量: {len(self.outer_polygons)}', 
                       transform=axes[2, 2].transAxes, fontsize=12)
        axes[2, 2].text(0.1, 0.8, f'内轮廓数量: {len(self.inner_polygons)}', 
                       transform=axes[2, 2].transAxes, fontsize=12)
        
        # 显示顶点详情
        y_pos = 0.7
        for i, poly in enumerate(self.outer_polygons):
            axes[2, 2].text(0.1, y_pos, f'外轮廓{i+1}: {poly["vertices"]}个顶点', 
                           transform=axes[2, 2].transAxes, fontsize=10, color='green')
            y_pos -= 0.08
        
        for i, poly in enumerate(self.inner_polygons):
            axes[2, 2].text(0.1, y_pos, f'内轮廓{i+1}: {poly["vertices"]}个顶点', 
                           transform=axes[2, 2].transAxes, fontsize=10, color='red')
            y_pos -= 0.08
        
        axes[2, 2].set_title('检测统计')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

    # ===== 连线检测相关方法 =====
    
    def is_line_inside_shape(self, point1, point2):
        """
        判断两个点之间的连线是否完全在图形内部（基于线段相交算法）
        
        核心逻辑：
        - 如果连线冲破外轮廓 → False
        - 如果连线冲破内轮廓 → False  
        - 否则 → True
        
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
        
        # 容忍度，处理数值精度问题
        tolerance = 3.0
        
        # 1. 检查是否冲破外轮廓
        for poly in self.outer_polygons:
            if self._line_breaks_through_outer_contour(p1, p2, poly['approx'], tolerance):
                return False
        
        # 2. 检查是否冲破内轮廓  
        for poly in self.inner_polygons:
            if self._line_breaks_through_inner_contour(p1, p2, poly['approx'], tolerance):
                return False
        
        # 3. 确保至少一个端点在有效区域内（带容忍度）
        if not (self._point_in_valid_area_with_tolerance(p1, tolerance) or 
                self._point_in_valid_area_with_tolerance(p2, tolerance)):
            return False
        
        return True

    def _point_in_valid_area_with_tolerance(self, point, tolerance=3.0):
        """
        检查点是否在有效区域内，带容忍度处理
        
        Args:
            point: (x, y) 坐标
            tolerance: 容忍度（像素）
            
        Returns:
            bool: 点是否在有效区域内
        """
        # 必须在至少一个外轮廓内（包括边界和容忍区域）
        in_outer = False
        for poly in self.outer_polygons:
            distance = cv2.pointPolygonTest(poly['approx'], point, True)
            if distance >= -tolerance:  # 包含边界和容忍区域
                in_outer = True
                break
        
        if not in_outer:
            return False
        
        # 不能在任何内轮廓内部（但边界和容忍区域可以接受）
        for poly in self.inner_polygons:
            distance = cv2.pointPolygonTest(poly['approx'], point, True)
            if distance > tolerance:  # 只排除明确在内轮廓内部的点
                return False
        
        return True

    def _line_breaks_through_outer_contour(self, p1, p2, contour, tolerance=3.0):
        """
        判断线段是否冲破外轮廓
        
        逻辑：如果线段与外轮廓相交，且有端点在外轮廓外部，则认为冲破
        
        Args:
            p1, p2: 线段端点
            contour: 外轮廓
            tolerance: 容忍度
            
        Returns:
            bool: 是否冲破外轮廓
        """
        # 检查端点是否在外轮廓外部
        p1_dist = cv2.pointPolygonTest(contour, p1, True)
        p2_dist = cv2.pointPolygonTest(contour, p2, True)
        
        p1_outside = p1_dist < -tolerance
        p2_outside = p2_dist < -tolerance
        
        # 如果两个端点都在外轮廓内部或边界上，不可能冲破
        if not (p1_outside or p2_outside):
            return False
        
        # 如果有端点在外部，检查是否与轮廓相交
        intersections = self._get_line_contour_intersections(p1, p2, contour)
        
        # 有相交且有端点在外部 → 冲破了外轮廓
        return len(intersections) > 0

    def _line_breaks_through_inner_contour(self, p1, p2, contour, tolerance=3.0):
        """
        判断线段是否冲破内轮廓
        
        逻辑：如果线段真正穿过内轮廓内部，则认为冲破
        
        Args:
            p1, p2: 线段端点
            contour: 内轮廓
            tolerance: 容忍度
            
        Returns:
            bool: 是否冲破内轮廓
        """
        # 检查线段与内轮廓的相交
        intersections = self._get_line_contour_intersections(p1, p2, contour)
        
        if len(intersections) == 0:
            return False  # 没有相交就不算冲破
        
        # 检查线段是否真正穿过内轮廓（而不仅仅是触碰边界）
        return self._line_crosses_inner_contour(p1, p2, contour, intersections, tolerance)

    def _line_crosses_inner_contour(self, p1, p2, contour, intersections, tolerance=3.0):
        """
        判断线段是否真正穿过内轮廓内部
        
        Args:
            p1, p2: 线段端点
            contour: 内轮廓
            intersections: 已计算的交点列表
            tolerance: 容忍度
            
        Returns:
            bool: 是否真正穿过内轮廓
        """
        if len(intersections) >= 2:
            # 多个交点，很可能穿过了内轮廓
            return True
        elif len(intersections) == 1:
            # 单个交点，检查线段中点是否在内轮廓内部
            mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid_dist = cv2.pointPolygonTest(contour, mid_point, True)
            return mid_dist > tolerance
        else:
            return False

    def _get_line_contour_intersections(self, p1, p2, contour):
        """
        计算线段与轮廓的所有交点
        
        Args:
            p1, p2: 线段端点
            contour: 轮廓
            
        Returns:
            list: 交点列表
        """
        intersections = []
        contour_points = contour.reshape(-1, 2)
        n_points = len(contour_points)
        
        for i in range(n_points):
            # 轮廓的当前边
            edge_p1 = tuple(contour_points[i])
            edge_p2 = tuple(contour_points[(i + 1) % n_points])
            
            # 检查线段与轮廓边的相交
            intersection = self._line_segments_intersect(p1, p2, edge_p1, edge_p2)
            if intersection is not None:
                intersections.append(intersection)
        
        return intersections

    def _line_segments_intersect(self, p1, p2, p3, p4):
        """
        计算两条线段的交点（如果存在）
        
        使用参数方程法：
        线段1: P = p1 + t*(p2-p1), t ∈ [0,1]
        线段2: P = p3 + u*(p4-p3), u ∈ [0,1]
        
        Args:
            p1, p2: 第一条线段的端点
            p3, p4: 第二条线段的端点
            
        Returns:
            tuple or None: 交点坐标 (x, y) 或 None（无交点）
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # 计算方向向量的叉积（分母）
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # 平行线或重合线
            return None
        
        # 计算参数 t 和 u
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # 检查交点是否在两条线段上
        if 0 <= t <= 1 and 0 <= u <= 1:
            # 计算交点坐标
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None

    def _point_in_valid_area(self, point):
        """
        检查点是否在有效区域内（保持向后兼容）
        """
        return self._point_in_valid_area_with_tolerance(point, tolerance=0.0)
    
    def check_line_by_vertex_indices(self, outer1_idx=None, vertex1_idx=None, 
                                    outer2_idx=None, vertex2_idx=None,
                                    inner1_idx=None, inner2_idx=None):
        """
        通过顶点序号检查连线是否在图形内部（简化版接口）
        
        Args:
            outer1_idx, vertex1_idx: 第一个点的外轮廓序号和顶点序号
            outer2_idx, vertex2_idx: 第二个点的外轮廓序号和顶点序号  
            inner1_idx, inner2_idx: 或者直接指定内轮廓的顶点序号
        
        Returns:
            bool: 连线是否在图形内部
        """
        # 构造点1
        if outer1_idx is not None and vertex1_idx is not None:
            point1 = {'contour_type': 'outer', 'contour_id': outer1_idx, 'vertex_id': vertex1_idx}
        elif inner1_idx is not None:
            point1 = {'contour_type': 'inner', 'contour_id': 0, 'vertex_id': inner1_idx}
        else:
            return False
        
        # 构造点2  
        if outer2_idx is not None and vertex2_idx is not None:
            point2 = {'contour_type': 'outer', 'contour_id': outer2_idx, 'vertex_id': vertex2_idx}
        elif inner2_idx is not None:
            point2 = {'contour_type': 'inner', 'contour_id': 0, 'vertex_id': inner2_idx}
        else:
            return False
        
        return self.is_line_inside_shape(point1, point2)

    def test_line_detection(self):
        """测试连线检测功能"""
        if not self.outer_polygons and not self.inner_polygons:
            print("请先进行多边形检测")
            return
        
        print("\n=== 连线检测测试 ===")
        
        # 方法1: 直接使用坐标
        result1 = self.is_line_inside_shape((100, 100), (200, 200))
        print(f"坐标连线 (100,100)-(200,200): {'在图形内' if result1 else '不在图形内'}")
        
        # 方法2: 使用顶点序号
        if len(self.outer_polygons) > 0 and len(self.outer_polygons[0]['approx']) > 1:
            point1 = {'contour_type': 'outer', 'contour_id': 0, 'vertex_id': 0}
            point2 = {'contour_type': 'outer', 'contour_id': 0, 'vertex_id': 1}
            result2 = self.is_line_inside_shape(point1, point2)
            print(f"外轮廓顶点连线: {'在图形内' if result2 else '不在图形内'}")
        
        # 方法3: 简化接口
        if len(self.outer_polygons) > 0:
            result3 = self.check_line_by_vertex_indices(outer1_idx=0, vertex1_idx=0,
                                                    outer2_idx=0, vertex2_idx=2)
            print(f"外轮廓0的顶点0-2连线: {'在图形内' if result3 else '不在图形内'}")


# ===== 使用示例和测试代码 =====

def test_polygon_detection():
    """测试多边形检测功能"""
    import os
    import glob
    
    # 创建多边形检测器实例
    detector = PolygonDetector(
        outer_vertex_merge_threshold=8.0,  # 外轮廓顶点合并阈值
        inner_vertex_merge_threshold=6.0,  # 内轮廓顶点合并阈值
        min_perimeter=50                    # 最小轮廓周长
    )
    
    print("开始多边形检测测试...")
    
    # 读取images/overlap/目录下的所有PNG图片
    image_dir = "images/overlap/"
    image_pattern = os.path.join(image_dir, "*.png")
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"在目录 {image_dir} 中未找到PNG图片文件")
        return None
    
    print(f"找到 {len(image_files)} 张图片")
    
    for i, image_path in enumerate(image_files):
        try:
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"无法读取图片: {image_path}")
                continue
            
            print(f"\n处理图片 {i + 1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # 检测多边形
            outer_polygons, inner_polygons = detector.detect_polygons(frame)
            
            # 打印检测结果摘要
            detector.print_detection_summary()
            
            # 获取顶点信息
            vertices_info = detector.get_vertices()
            print(f"\n顶点信息:")
            print(f"外轮廓顶点: {len(vertices_info['outer_vertices'])} 个形状")
            print(f"内轮廓顶点: {len(vertices_info['inner_vertices'])} 个形状")
            
            # 可视化结果
            detector.visualize_results(f'图片 {i+1}: {os.path.basename(image_path)}')
            
            # 返回检测器实例以便进一步测试
            return detector
            
        except Exception as e:
            print(f"处理图片时发生错误: {e}")
            continue
    
    print("测试结束")
    return None


def test_line_detection_comprehensive(detector):
    """全面测试连线检测功能 - 重点测试90度角顶点之间的连线"""
    print("\n" + "="*50)
    print("开始连线检测测试...")
    print("="*50)
    
    if not detector.outer_polygons and not detector.inner_polygons:
        print("请先进行多边形检测")
        return
    
    # 1. 获取90度角顶点
    valid_vertices = detector.get_90_degree_vertices(angle_tolerance=10.0)
    
    if not valid_vertices:
        print("未找到90度角顶点，跳过连线测试")
        return
    
    print(f"\n=== 90度角顶点连线测试 ===")
    print(f"找到 {len(valid_vertices)} 个有效顶点，开始测试连线...")
    
    # 2. 测试90度角顶点之间的连线
    test_count = 0
    inside_count = 0
    
    for i in range(len(valid_vertices)):
        for j in range(i + 1, len(valid_vertices)):
            vertex1 = valid_vertices[i]
            vertex2 = valid_vertices[j]
            
            # 构造连线测试
            point1 = {
                'contour_type': 'outer', 
                'contour_id': vertex1['contour_id'], 
                'vertex_id': vertex1['vertex_id']
            }
            point2 = {
                'contour_type': 'outer', 
                'contour_id': vertex2['contour_id'], 
                'vertex_id': vertex2['vertex_id']
            }
            
            result = detector.is_line_inside_shape(point1, point2)
            test_count += 1
            if result:
                inside_count += 1
            
            # 输出测试结果
            coord1 = vertex1['coordinates']
            coord2 = vertex2['coordinates']
            angle1 = vertex1['angle']
            angle2 = vertex2['angle']
            
            print(f"  连线 {i+1}-{j+1}: "
                  f"轮廓{vertex1['contour_id']}顶点{vertex1['vertex_id']}({coord1[0]},{coord1[1]},{angle1:.1f}°) - "
                  f"轮廓{vertex2['contour_id']}顶点{vertex2['vertex_id']}({coord2[0]},{coord2[1]},{angle2:.1f}°) "
                  f"{'✓在图形内' if result else '✗不在图形内'}")
    
    print(f"\n90度角顶点连线测试结果:")
    print(f"  总测试数: {test_count}")
    print(f"  在图形内: {inside_count} ({inside_count/test_count*100:.1f}%)" if test_count > 0 else "  无测试数据")
    print(f"  在图形外: {test_count - inside_count} ({(test_count-inside_count)/test_count*100:.1f}%)" if test_count > 0 else "")
    
    # 3. 额外测试：90度角顶点与其相邻顶点的连线（应该都在图形内）
    print(f"\n=== 90度角顶点与相邻顶点连线测试 ===")
    
    for i, vertex in enumerate(valid_vertices):
        contour_id = vertex['contour_id']
        vertex_id = vertex['vertex_id']
        
        if contour_id < len(detector.outer_polygons):
            n_vertices = len(detector.outer_polygons[contour_id]['approx'])
            
            # 测试与前一个顶点的连线
            prev_vertex_id = (vertex_id - 1) % n_vertices
            result_prev = detector.check_line_by_vertex_indices(
                outer1_idx=contour_id, vertex1_idx=vertex_id,
                outer2_idx=contour_id, vertex2_idx=prev_vertex_id
            )
            
            # 测试与后一个顶点的连线
            next_vertex_id = (vertex_id + 1) % n_vertices
            result_next = detector.check_line_by_vertex_indices(
                outer1_idx=contour_id, vertex1_idx=vertex_id,
                outer2_idx=contour_id, vertex2_idx=next_vertex_id
            )
            
            coord = vertex['coordinates']
            angle = vertex['angle']
            
            print(f"  90度顶点 {i+1} (轮廓{contour_id}顶点{vertex_id}, {coord}, {angle:.1f}°):")
            print(f"    -> 前一顶点{prev_vertex_id}: {'✓在图形内' if result_prev else '✗不在图形内'}")
            print(f"    -> 后一顶点{next_vertex_id}: {'✓在图形内' if result_next else '✗不在图形内'}")
    
    print("\n连线检测测试完成！")


# 主程序入口
if __name__ == "__main__":
    print("多边形检测与连线测试程序")
    print("="*50)
    
    # 先进行多边形检测
    detector = test_polygon_detection()
    
    if detector:
        # 如果检测成功，进行连线测试
        test_line_detection_comprehensive(detector)
    else:
        print("多边形检测失败，无法进行连线测试")
    
    print("\n所有测试完成！")