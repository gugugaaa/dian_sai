import sys
import os
# 添加根目录到路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import cv2
import numpy as np
from system_initializer import MeasurementSystem

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
    print("初始化测量系统...")
    
    try:
        system = MeasurementSystem("calib.yaml", 500)
        print("系统初始化成功")
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return
    
    print("开始重合正方形检测测试...")
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
                cv2.imshow("Shape Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # 预处理
            edges = system.preprocessor.preprocess(cropped_frame)
            
            # 检测A4纸边框
            ok, corners = system.border_detector.detect_border(edges, cropped_frame)
            if not ok:
                print("无法检测A4边框")
                cv2.imshow("Shape Detection", cropped_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # 后裁切
            post_cropped_frame, adjusted_corners = system.border_detector.post_crop(cropped_frame, corners, inset_pixels=5)
            
            # --- 在这里测试处理函数 ---
            edges_raw, edge_debug, poly_debug, polygons = get_poly(post_cropped_frame)
            # --- ---
            
            # 显示调试图像
            cv2.imshow("Canny Edges", edges_raw)
            cv2.imshow("Edge Detection", edge_debug)
            cv2.imshow("Polygon Fitting", poly_debug)
            
            # 打印检测到的多边形信息
            if polygons:
                print(f"检测到 {len(polygons)} 个多边形:")
                for i, poly in enumerate(polygons):
                    print(f"  多边形 {i+1}: {poly['vertices']} 个顶点, 周长 {poly['perimeter']:.1f}")
            
            # 检测重合正方形
            squares = detect_overlapping_squares(post_cropped_frame)
            
            # 创建结果显示图像
            result_frame = post_cropped_frame.copy()
            
            if squares:
                print(f"检测到 {len(squares)} 个正方形:")
                for i, square in enumerate(squares):
                    side_length = square['side_length']
                    print(f"  正方形 {i+1}: 边长 {side_length:.1f} 像素")
                
                # 绘制检测结果
                result_frame = draw_overlapping_squares(result_frame, squares)
            else:
                print("未检测到正方形")
            
            # 显示结果
            cv2.imshow("Shape Detection", result_frame)
            
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
    test_shape_detection()