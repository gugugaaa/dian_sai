import cv2
import numpy as np
import math
import random

class GeometryDrawer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
    def clear_canvas(self):
        """清空画布"""
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
    
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
    
    def create_complex_polygon(self, center_x=400, center_y=300):
        """
        创建复杂多边形：由2个轴对称和2个随机旋转的不同大小正方形重叠形成
        """
        polygons = []
        
        # 正方形1: 轴对称（水平对称）- 放大尺寸
        size1 = 200
        square1 = [
            (center_x - size1//2, center_y - size1//2),
            (center_x + size1//2, center_y - size1//2),
            (center_x + size1//2, center_y + size1//2),
            (center_x - size1//2, center_y + size1//2)
        ]
        polygons.append(square1)
        
        # 正方形2: 轴对称（垂直对称）- 放大尺寸
        size2 = 160
        offset_x = 120
        square2 = [
            (center_x - size2//2 + offset_x, center_y - size2//2),
            (center_x + size2//2 + offset_x, center_y - size2//2),
            (center_x + size2//2 + offset_x, center_y + size2//2),
            (center_x - size2//2 + offset_x, center_y + size2//2)
        ]
        polygons.append(square2)
        
        # 正方形3: 随机旋转 - 放大尺寸
        size3 = 140
        angle3 = random.uniform(0, 2 * math.pi)
        offset_x3 = random.uniform(-60, 60)
        offset_y3 = random.uniform(-60, 60)
        
        square3_center_x = center_x + offset_x3
        square3_center_y = center_y + offset_y3
        
        square3 = self.rotate_square(
            [(square3_center_x - size3//2, square3_center_y - size3//2),
             (square3_center_x + size3//2, square3_center_y - size3//2),
             (square3_center_x + size3//2, square3_center_y + size3//2),
             (square3_center_x - size3//2, square3_center_y + size3//2)],
            angle3, (square3_center_x, square3_center_y)
        )
        polygons.append(square3)
        
        # 正方形4: 随机旋转 - 放大尺寸
        size4 = 120
        angle4 = random.uniform(0, 2 * math.pi)
        offset_x4 = random.uniform(-100, 100)
        offset_y4 = random.uniform(-100, 100)
        
        square4_center_x = center_x + offset_x4
        square4_center_y = center_y + offset_y4
        
        square4 = self.rotate_square(
            [(square4_center_x - size4//2, square4_center_y - size4//2),
             (square4_center_x + size4//2, square4_center_y - size4//2),
             (square4_center_x + size4//2, square4_center_y + size4//2),
             (square4_center_x - size4//2, square4_center_y + size4//2)],
            angle4, (square4_center_x, square4_center_y)
        )
        polygons.append(square4)
        
        return polygons
    
    def rotate_square(self, square, angle, center):
        """旋转正方形"""
        cx, cy = center
        rotated_square = []
        
        for x, y in square:
            # 平移到原点
            x_shifted = x - cx
            y_shifted = y - cy
            
            # 旋转
            x_rotated = x_shifted * math.cos(angle) - y_shifted * math.sin(angle)
            y_rotated = x_shifted * math.sin(angle) + y_shifted * math.cos(angle)
            
            # 平移回去
            x_final = x_rotated + cx
            y_final = y_rotated + cy
            
            rotated_square.append((x_final, y_final))
        
        return rotated_square
    
    def point_in_polygon_with_tolerance(self, point, polygon_contour, tolerance=5.0):
        """
        检查点是否在多边形内部，带容忍度处理
        参数:
        - point: 检查的点 (x, y)
        - polygon_contour: 多边形轮廓点数组
        - tolerance: 容忍度，负值表示向内扩张，正值表示向外扩张
        
        返回: True如果点在多边形内部（含容忍度），False否则
        """
        # 将轮廓转换为正确的格式
        if isinstance(polygon_contour, list):
            contour = np.array([(int(x), int(y)) for x, y in polygon_contour], dtype=np.int32)
        else:
            contour = polygon_contour
        
        # 使用cv2.pointPolygonTest计算点到多边形的距离
        # 返回值：正值表示在内部，负值表示在外部，0表示在边界上
        distance = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
        
        # 如果距离大于等于负容忍度，则认为点在多边形内部
        return distance >= -tolerance
    
    def square_in_polygon(self, square, polygon_contour, tolerance=5.0):
        """
        检测正方形是否在多边形内部
        算法：检查正方形的四个角是否都在多边形内部（包括边缘）
        
        参数:
        - square: 正方形的四个顶点列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        - polygon_contour: 多边形轮廓
        - tolerance: 容忍度
        
        返回: (is_inside, corner_results)
        - is_inside: 布尔值，表示是否所有角都在多边形内
        - corner_results: 每个角的检测结果列表
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
    
    def draw_line_segment(self, p1, p2, color=(255, 255, 255), thickness=2):
        """绘制线段"""
        cv2.line(self.canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    
    def draw_polygon(self, points, color=(0, 255, 0), thickness=2, fill=False):
        """绘制多边形"""
        points_int = np.array([(int(x), int(y)) for x, y in points], np.int32)
        points_int = points_int.reshape((-1, 1, 2))
        
        if fill:
            cv2.fillPoly(self.canvas, [points_int], color)
        else:
            cv2.polylines(self.canvas, [points_int], True, color, thickness)
    
    def draw_squares(self, squares, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], thickness=2):
        """绘制三个正方形，可以指定不同颜色"""
        for i, square in enumerate(squares):
            color = colors[i % len(colors)]
            self.draw_polygon(square, color, thickness=thickness)
    
    def draw_complex_polygon(self, polygons, color=(0, 100, 255)):
        """绘制复杂多边形的外轮廓"""
        outline = self.compute_union_outline(polygons)
        if len(outline) > 0:
            self.draw_polygon(outline, color, thickness=3)  # 加粗轮廓
        return outline
    
    def compute_union_outline(self, polygons):
        """计算多个多边形的并集外轮廓"""
        # 创建临时图像用于计算轮廓
        temp_img = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 将所有多边形填充到临时图像中
        for polygon in polygons:
            points_int = np.array([(int(x), int(y)) for x, y in polygon], np.int32)
            cv2.fillPoly(temp_img, [points_int], 255)
        
        # 查找轮廓
        contours, _ = cv2.findContours(temp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 返回最大的轮廓（外轮廓）
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour.reshape(-1, 2)
        return []
    
    def add_text(self, text, position, color=(0, 0, 0), font_scale=0.6):
        """在画布上添加文字"""
        cv2.putText(self.canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    
    def show_canvas(self, window_name="Geometry Visualization"):
        """显示画布"""
        cv2.imshow(window_name, self.canvas)
    
    def draw_circle(self, center, radius=5, color=(255, 0, 255), thickness=-1):
        """绘制圆点"""
        cv2.circle(self.canvas, (int(center[0]), int(center[1])), radius, color, thickness)
    
    def draw_corner_indicators(self, square, corner_results, base_color):
        """绘制正方形角点的检测结果指示器"""
        for result in corner_results:
            corner = result['position']
            is_inside = result['is_inside']
            
            # 根据检测结果选择颜色
            if is_inside:
                color = (0, 255, 0)  # 绿色表示在内部
                radius = 4
            else:
                color = (0, 0, 255)  # 红色表示在外部
                radius = 6
            
            self.draw_circle(corner, radius=radius, color=color, thickness=-1)

def main():
    """主函数 - 测试正方形在多边形内部检测算法"""
    print("几何算法测试环境启动...")
    print("测试算法：如果构造的正方形的四个角都在多边形里面，那么这个正方形在多边形里面")
    print("按任意键退出程序")
    
    # 设置随机种子以便重现结果
    random.seed(42)
    
    # 初始化绘制器
    drawer = GeometryDrawer()
    
    # 创建复杂多边形
    complex_polygons = drawer.create_complex_polygon()
    
    # 测试多个不同的线段
    test_segments = [
        # 测试1: 多边形内部的线段
        ((350, 280), (450, 320)),
        # 测试2: 跨越边界的线段  
        ((200, 200), (300, 250)),
        # 测试3: 完全在外部的线段
        ((100, 100), (150, 120))
    ]
    
    tolerance = 5.0  # 容忍度设置
    
    for test_idx, (p1, p2) in enumerate(test_segments):
        print(f"\n=== 测试 {test_idx + 1} ===")
        print(f"线段端点: {p1} -> {p2}")
        
        # 清空画布
        drawer.clear_canvas()
        
        # 绘制复杂多边形并获取轮廓
        outline = drawer.draw_complex_polygon(complex_polygons)
        
        if len(outline) == 0:
            print("警告: 无法获取多边形轮廓")
            continue
        
        # 创建三个正方形
        squares = drawer.create_squares_from_segment(p1, p2)
        
        # 绘制线段
        drawer.draw_line_segment(p1, p2, (0, 0, 0), 3)
        
        # 绘制线段端点
        drawer.draw_circle(p1, radius=8, color=(255, 0, 255))
        drawer.draw_circle(p2, radius=8, color=(255, 0, 255))
        
        # 测试每个正方形
        square_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红、绿、蓝
        square_names = ["Square 1 (Side-based 1)", "Square 2 (Side-based 2)", "Square 3 (Diagonal-based)"]
        
        results_text = []
        
        for i, square in enumerate(squares):
            # 检测正方形是否在多边形内部
            is_inside, corner_results = drawer.square_in_polygon(square, outline, tolerance)
            
            # 根据检测结果选择绘制颜色和样式
            if is_inside:
                # 在内部 - 用实线绘制
                drawer.draw_polygon(square, square_colors[i], thickness=3)
                result_text = "INSIDE"
            else:
                # 不完全在内部 - 用虚线效果绘制（通过较细的线模拟）
                drawer.draw_polygon(square, square_colors[i], thickness=1)
                result_text = "OUTSIDE"
            
            # 绘制角点指示器
            drawer.draw_corner_indicators(square, corner_results, square_colors[i])
            
            # 打印详细结果
            print(f"{square_names[i]}: {result_text}")
            for j, corner_result in enumerate(corner_results):
                status = "✓" if corner_result['is_inside'] else "✗"
                print(f"  Corner {j+1}: {status}")
            
            results_text.append(f"{square_names[i][:8]}: {result_text}")
        
        # 添加说明文字
        y_offset = 30
        drawer.add_text(f"Test {test_idx + 1}: Segment ({int(p1[0])},{int(p1[1])}) -> ({int(p2[0])},{int(p2[1])})", 
                       (10, y_offset))
        
        for i, result_text in enumerate(results_text):
            y_offset += 25
            color = (0, 150, 0) if "INSIDE" in result_text else (0, 0, 150)
            drawer.add_text(result_text, (10, y_offset), color)
        
        y_offset += 30
        drawer.add_text("Green dots: corners inside polygon", (10, y_offset), (0, 150, 0))
        y_offset += 20
        drawer.add_text("Red dots: corners outside polygon", (10, y_offset), (0, 0, 150))
        y_offset += 20
        drawer.add_text(f"Tolerance: {tolerance} pixels", (10, y_offset))
        
        # 在底部添加操作提示
        drawer.add_text("Press any key for next test, ESC to exit", 
                       (10, drawer.height - 10), (100, 100, 100))
        
        # 显示结果
        drawer.show_canvas(f"Algorithm Test {test_idx + 1}")
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC键退出
            break
    
    cv2.destroyAllWindows()
    print("\n测试完成！")
    
if __name__ == "__main__":
    main()