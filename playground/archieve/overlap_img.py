import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def detect_overlapping_squares(frame):
    """检测部分重合的轴对齐正方形 - 占位函数"""
    # TODO: 实现重合正方形检测逻辑
    # 开发说明：！先不要写进来！，在外面定义处理方法，直接在主循环里面调用，这样就可以返回中间图像，方便调试
    return []

# --- 我们首先测试这种方法 ---
def get_poly(frame):

    """检测边缘并进行多边形拟合"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # OTSU阈值处理
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 边缘检测
    edges = cv2.Canny(otsu_thresh, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建调试图像
    edge_debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    poly_debug = frame.copy()
    
    polygons = []
    
    for contour in contours:
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 跳过过小的轮廓
        if perimeter < 100:
            continue
            
        # 多边形拟合
        epsilon = 0.005 * perimeter  # 拟合精度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 绘制原始轮廓到边缘调试图
        cv2.drawContours(edge_debug, [contour], -1, (0, 255, 0), 2)
        
        # 绘制拟合多边形到多边形调试图
        cv2.drawContours(poly_debug, [approx], -1, (0, 255, 0), 2)
        
        # 标注顶点数量
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(poly_debug, f"V:{len(approx)}", (cx-20, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制顶点
        for point in approx:
            cv2.circle(poly_debug, tuple(point[0]), 5, (0, 0, 255), -1)
        
        polygons.append({
            'contour': contour,
            'approx': approx,
            'vertices': len(approx),
            'perimeter': perimeter
        })
    
    return edges, edge_debug, poly_debug, polygons
# --- 我们测试的方法结束 ---

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
    print("开始重合正方形检测测试...")
    
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
            
            print(f"处理图片 {i + 1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # --- 在这里测试处理函数 ---
            edges_raw, edge_debug, poly_debug, polygons = get_poly(frame)
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
            
            # 打印检测到的多边形信息
            if polygons:
                print(f"检测到 {len(polygons)} 个多边形:")
                for j, poly in enumerate(polygons):
                    print(f"  多边形 {j+1}: {poly['vertices']} 个顶点, 周长 {poly['perimeter']:.1f}")
            
            # 使用matplotlib显示所有图像
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'图片 {i+1}/{len(image_files)}: {os.path.basename(image_path)}', fontsize=16)
            
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
            axes[0, 2].set_title('轮廓检测')
            axes[0, 2].axis('off')
            
            # 多边形拟合调试图 (BGR转RGB)
            axes[1, 0].imshow(cv2.cvtColor(poly_debug, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('多边形拟合')
            axes[1, 0].axis('off')
            
            # 形状检测结果 (BGR转RGB)
            axes[1, 1].imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('形状检测结果')
            axes[1, 1].axis('off')
            
            # 隐藏最后一个子图
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
                
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            continue
    
    print("测试结束")


if __name__ == "__main__":
    test_shape_detection()