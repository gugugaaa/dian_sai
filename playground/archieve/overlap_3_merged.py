import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def merge_close_vertices(vertices, distance_threshold=5.0):
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

def detect_overlapping_squares(frame):
    """检测部分重合的轴对齐正方形 - 占位函数"""
    # TODO: 实现重合正方形检测逻辑
    # 开发说明：！先不要写进来！，在外面定义处理方法，直接在主循环里面调用，这样就可以返回中间图像，方便调试
    return []

# --- 改进的内外轮廓检测方法 ---
def get_poly_with_hierarchy(frame):
    """检测边缘并进行多边形拟合，包含内外轮廓检测，过滤重复轮廓，并合并相近顶点"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # OTSU阈值处理
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 边缘检测
    edges = cv2.Canny(otsu_thresh, 50, 150)
    
    # 查找轮廓，包含层次结构信息
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建调试图像
    edge_debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    poly_debug = frame.copy()
    hierarchy_debug = frame.copy()
    filtered_debug = frame.copy()
    vertex_merge_debug = frame.copy()  # 新增：顶点合并后的调试图
    
    # 分别存储所有轮廓和过滤后的轮廓
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
            if perimeter < 50:  # 降低阈值以便检测小的内轮廓
                continue
            
            # 分析层次结构
            # hierarchy[i] = [next, previous, first_child, parent]
            next_contour, prev_contour, first_child, parent = hierarchy[i]
            
            # 判断是外轮廓还是内轮廓
            is_outer = parent == -1  # 没有父轮廓的是外轮廓
            is_inner = parent != -1  # 有父轮廓的是内轮廓
            
            # 根据内外轮廓使用不同的拟合精度
            if is_outer:
                epsilon = 0.005 * perimeter  # 外轮廓使用高精度
            else:
                epsilon = 0.02 * perimeter   # 内轮廓使用低精度
            
            # 多边形拟合
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 绘制到原始调试图（显示所有轮廓）
            color = (0, 255, 0) if is_outer else (255, 0, 0)  # 外轮廓绿色，内轮廓红色
            cv2.drawContours(edge_debug, [contour], -1, color, 2)
            cv2.drawContours(poly_debug, [approx], -1, color, 2)
            
            # 计算轮廓中心用于标注
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 标注信息：类型和顶点数
                label = f"{'外' if is_outer else '内'}:{len(approx)}"
                cv2.putText(poly_debug, label, (cx-20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 在层次结构调试图上标注更详细信息
                detail_label = f"ID:{i} P:{parent}"
                cv2.putText(hierarchy_debug, detail_label, (cx-30, cy-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 绘制顶点
            for point in approx:
                cv2.circle(poly_debug, tuple(point[0]), 3, color, -1)
            
            # 存储所有轮廓信息
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
        
        # 第二步：过滤重复轮廓
        def is_similar_contour(poly1, poly2, perimeter_threshold=0.1, vertex_threshold=2):
            """判断两个轮廓是否相似（可能是同一形状的内外边缘）"""
            # 周长相似度检查
            perimeter_ratio = abs(poly1['perimeter'] - poly2['perimeter']) / max(poly1['perimeter'], poly2['perimeter'])
            
            # 顶点数相似度检查
            vertex_diff = abs(poly1['vertices'] - poly2['vertices'])
            
            return perimeter_ratio < perimeter_threshold and vertex_diff <= vertex_threshold
        
        # 分类并过滤轮廓
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
                if parent_poly and is_similar_contour(poly, parent_poly):
                    # 跳过与父轮廓相似的内轮廓（重复轮廓）
                    print(f"过滤重复轮廓: 内轮廓{poly['id']} 与父轮廓{parent_id} 相似")
                    continue
                else:
                    # 保留真正的内轮廓
                    inner_polygons.append(poly)
        
        # 第三步：对所有多边形进行顶点合并处理
        print("\n=== 开始顶点合并处理 ===")
        
        # 处理外轮廓
        for i, poly in enumerate(outer_polygons):
            print(f"\n处理外轮廓 {i+1}:")
            original_vertices = poly['approx']
            merged_vertices = merge_close_vertices(original_vertices, distance_threshold=8.0)
            poly['approx'] = merged_vertices
            poly['vertices'] = len(merged_vertices)
        
        # 处理内轮廓
        for i, poly in enumerate(inner_polygons):
            print(f"\n处理内轮廓 {i+1}:")
            original_vertices = poly['approx']
            merged_vertices = merge_close_vertices(original_vertices, distance_threshold=6.0)
            poly['approx'] = merged_vertices
            poly['vertices'] = len(merged_vertices)
        
        # 绘制过滤后的结果（顶点合并前）
        for poly in outer_polygons:
            cv2.drawContours(filtered_debug, [poly['approx']], -1, (0, 255, 0), 3)
            
            M = cv2.moments(poly['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(filtered_debug, f"外:{poly['vertices']}", (cx-20, cy-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for poly in inner_polygons:
            cv2.drawContours(filtered_debug, [poly['approx']], -1, (0, 0, 255), 2)
            
            M = cv2.moments(poly['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(filtered_debug, f"内:{poly['vertices']}", (cx-20, cy+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制顶点合并后的结果
        for poly in outer_polygons:
            cv2.drawContours(vertex_merge_debug, [poly['approx']], -1, (0, 255, 0), 3)
            
            # 绘制合并后的顶点（用较大的圆圈标识）
            for point in poly['approx']:
                cv2.circle(vertex_merge_debug, tuple(point[0]), 5, (0, 255, 0), -1)
                cv2.circle(vertex_merge_debug, tuple(point[0]), 7, (255, 255, 255), 2)
            
            M = cv2.moments(poly['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vertex_merge_debug, f"外:{poly['vertices']}", (cx-20, cy-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for poly in inner_polygons:
            cv2.drawContours(vertex_merge_debug, [poly['approx']], -1, (0, 0, 255), 2)
            
            # 绘制合并后的顶点
            for point in poly['approx']:
                cv2.circle(vertex_merge_debug, tuple(point[0]), 4, (0, 0, 255), -1)
                cv2.circle(vertex_merge_debug, tuple(point[0]), 6, (255, 255, 255), 2)
            
            M = cv2.moments(poly['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vertex_merge_debug, f"内:{poly['vertices']}", (cx-20, cy+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return edges, edge_debug, poly_debug, hierarchy_debug, filtered_debug, vertex_merge_debug, outer_polygons, inner_polygons

# --- 融合显示内外轮廓的函数 ---
def create_combined_contour_display(frame, outer_polygons, inner_polygons):
    """创建融合的内外轮廓显示"""
    combined_display = frame.copy()
    
    # 绘制外轮廓（绿色，较粗线条）
    for i, poly in enumerate(outer_polygons):
        cv2.drawContours(combined_display, [poly['approx']], -1, (0, 255, 0), 3)
        
        # 标注外轮廓序号
        M = cv2.moments(poly['contour'])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(combined_display, f"外{i+1}", (cx-15, cy-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制内轮廓（红色，较细线条）
    for i, poly in enumerate(inner_polygons):
        cv2.drawContours(combined_display, [poly['approx']], -1, (0, 0, 255), 2)
        
        # 标注内轮廓序号和所属外轮廓
        M = cv2.moments(poly['contour'])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(combined_display, f"内{i+1}", (cx-15, cy+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return combined_display

def draw_overlapping_squares(frame, squares):
    """绘制检测到的重合正方形"""
    result = frame.copy()
    
    for i, square in enumerate(squares):
        # 绘制轮廓
        cv2.drawContours(result, [square['contour']], -1, (0, 255, 0), 2)
        
        # 绘制边界框
        x, y, w, h = square['bbox']
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # 标注序号
        cv2.putText(result, f"{i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result

def test_shape_detection():
    """测试形状检测功能"""
    print("开始重合正方形检测测试（包含内外轮廓和顶点合并）...")
    
    # 读取images/overlap/目录下的所有PNG图片
    image_dir = "images/overlap/"
    image_pattern = os.path.join(image_dir, "*.png")
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"在目录 {image_dir} 中未找到PNG图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print("关闭当前图片窗口查看下一张图片")
    
    for i, image_path in enumerate(image_files):
        try:
            # 读取当前图片
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"无法读取图片: {image_path}")
                continue
            
            print(f"\n处理图片 {i + 1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # --- 使用改进的内外轮廓检测函数（包含顶点合并） ---
            edges_raw, edge_debug, poly_debug, hierarchy_debug, filtered_debug, vertex_merge_debug, outer_polygons, inner_polygons = get_poly_with_hierarchy(frame)
            
            # 创建融合显示
            combined_display = create_combined_contour_display(frame, outer_polygons, inner_polygons)
            # --- ---
            
            # 检测重合正方形
            squares = detect_overlapping_squares(frame)
            
            # 创建结果显示图像
            result_frame = frame.copy()
            
            if squares:
                print(f"检测到 {len(squares)} 个正方形:")
                for j, square in enumerate(squares):
                    side_length = square['side_length']
                    print(f"  正方形 {j+1}: 边长 {side_length:.1f} 像素")
                
                # 绘制检测结果
                result_frame = draw_overlapping_squares(result_frame, squares)
            else:
                print("未检测到正方形")
            
            # 打印检测到的轮廓信息
            print(f"\n过滤后检测到 {len(outer_polygons)} 个外轮廓:")
            for j, poly in enumerate(outer_polygons):
                print(f"  外轮廓 {j+1}: {poly['vertices']} 个顶点, 周长 {poly['perimeter']:.1f}, "
                      f"面积 {poly['area']:.1f}, 有子轮廓: {'是' if poly['has_children'] else '否'}")
            
            print(f"过滤后检测到 {len(inner_polygons)} 个内轮廓:")
            for j, poly in enumerate(inner_polygons):
                print(f"  内轮廓 {j+1}: {poly['vertices']} 个顶点, 周长 {poly['perimeter']:.1f}, "
                      f"面积 {poly['area']:.1f}, 父轮廓ID: {poly['parent']}")
            
            # 使用matplotlib显示所有图像（现在是3x3布局）
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            fig.suptitle(f'图片 {i+1}/{len(image_files)}: {os.path.basename(image_path)}', fontsize=16)
            
            # 第一行
            # 原始图片 (BGR转RGB)
            axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('原始图片')
            axes[0, 0].axis('off')
            
            # Canny边缘检测 (灰度图)
            axes[0, 1].imshow(edges_raw, cmap='gray')
            axes[0, 1].set_title('Canny边缘检测')
            axes[0, 1].axis('off')
            
            # 边缘检测调试图 (BGR转RGB)
            axes[0, 2].imshow(cv2.cvtColor(edge_debug, cv2.COLOR_BGR2RGB))
            axes[0, 2].set_title('所有轮廓检测\n(绿色:外轮廓 红色:内轮廓)')
            axes[0, 2].axis('off')
            
            # 第二行
            # 多边形拟合调试图 (BGR转RGB)
            axes[1, 0].imshow(cv2.cvtColor(poly_debug, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('多边形拟合')
            axes[1, 0].axis('off')
            
            # 层次结构调试图 (BGR转RGB)
            axes[1, 1].imshow(cv2.cvtColor(hierarchy_debug, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('层次结构')
            axes[1, 1].axis('off')
            
            # 过滤后的结果 (BGR转RGB)
            axes[1, 2].imshow(cv2.cvtColor(filtered_debug, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title('过滤后轮廓\n(去除重复轮廓)')
            axes[1, 2].axis('off')
            
            # 第三行
            # 顶点合并后的结果 (BGR转RGB) - 新增
            axes[2, 0].imshow(cv2.cvtColor(vertex_merge_debug, cv2.COLOR_BGR2RGB))
            axes[2, 0].set_title('顶点合并后\n(白圈标识合并顶点)')
            axes[2, 0].axis('off')
            
            # 融合显示 (BGR转RGB)
            axes[2, 1].imshow(cv2.cvtColor(combined_display, cv2.COLOR_BGR2RGB))
            axes[2, 1].set_title('内外轮廓融合显示')
            axes[2, 1].axis('off')
            
            # 形状检测结果 (BGR转RGB)
            axes[2, 2].imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            axes[2, 2].set_title('形状检测结果')
            axes[2, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
                
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            continue
    
    print("测试结束")


if __name__ == "__main__":
    test_shape_detection()