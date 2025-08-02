import cv2
import numpy as np
from moving_average_filter import MovingAverageFilter

class SingleShapeDetectionAlgorithm:
    """单个形状检测算法类 - 检测圆形、三角形、正方形"""
    
    def __init__(self, measurement_system, **kwargs):
        """
        初始化单个形状检测算法
        
        Args:
            measurement_system: 测量系统实例
            **kwargs: 可调参数
                - filter_window_size: 移动平均值窗口大小 (默认: 10)
                - enable_size_filtering: 是否启用尺寸过滤 (默认: True)
                - min_shape_size_cm: 最小形状尺寸(cm) (默认: 1.0)
                - max_shape_size_cm: 最大形状尺寸(cm) (默认: 20.0)
                - post_crop_inset: 后裁剪内边距像素 (默认: 5)
        """
        
        # 存储测量系统引用
        self.measurement_system = measurement_system
        self.shape_detector = measurement_system.shape_detector
        
        # 可调参数
        self.config = {
            'filter_window_size': kwargs.get('filter_window_size', 10),
            'enable_size_filtering': kwargs.get('enable_size_filtering', True),
            'min_shape_size_cm': kwargs.get('min_shape_size_cm', 1.0),
            'max_shape_size_cm': kwargs.get('max_shape_size_cm', 20.0),
            'post_crop_inset': kwargs.get('post_crop_inset', 5)
        }
        
        # 初始化组件
        try:
            self.avg_filter = MovingAverageFilter(window_size=self.config['filter_window_size'])
            
            print("单个形状检测算法初始化成功")
            print(f"配置参数: {self.config}")
            
        except Exception as e:
            raise RuntimeError(f"单个形状检测算法初始化失败: {e}")
    
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
        处理post_crop后的图像，检测单个形状
        
        Args:
            post_cropped_frame: post_crop后的图像
            adjusted_corners: 调整后的A4纸角点
            D_corrected: 修正后的距离
            K: 相机内参矩阵
            
        Returns:
            dict: 包含检测结果的字典
        """
        try:
            # 1. 检测形状
            shape, x_pix, shape_params = self.shape_detector.detect_shape(post_cropped_frame)
            
            if not shape:
                return {
                    'shape': {
                        'type': None,
                        'params': None,
                        'size_cm': None,
                        'size_cm_raw': None,
                        'size_pixels': None,
                        'detection_success': False,
                        'filtered_out': False,
                        'filter_reason': None
                    },
                    'statistics': {
                        'size_filtered': None,
                        'distance_filtered': D_corrected,
                        'filter_samples': len(self.avg_filter.size_history)
                    },
                    'success': True,
                    'message': 'no shape detected'
                }
            
            # 2. 计算实际尺寸
            x_cm_raw = self.shape_detector.calculate_X(x_pix, D_corrected, K, adjusted_corners)
            
            # 3. 应用尺寸过滤
            filtered_out = False
            filter_reason = None
            
            if self.config['enable_size_filtering']:
                if (x_cm_raw < self.config['min_shape_size_cm'] or 
                    x_cm_raw > self.config['max_shape_size_cm']):
                    filtered_out = True
                    filter_reason = f"尺寸超出范围 [{self.config['min_shape_size_cm']}, {self.config['max_shape_size_cm']}]cm"
            
            # 4. 应用移动平均值滤波器
            x_cm_filtered, D_filtered = None, D_corrected
            if not filtered_out:
                x_cm_filtered, D_filtered = self.avg_filter.update(x_cm_raw, D_corrected)
            
            # 5. 构建返回结果
            result = {
                'shape': {
                    'type': shape,
                    'params': shape_params,
                    'size_cm': x_cm_filtered if not filtered_out else x_cm_raw,
                    'size_cm_raw': x_cm_raw,
                    'size_pixels': x_pix,
                    'detection_success': True,
                    'filtered_out': filtered_out,
                    'filter_reason': filter_reason
                },
                'statistics': {
                    'size_filtered': x_cm_filtered,
                    'distance_filtered': D_filtered,
                    'filter_samples': len(self.avg_filter.size_history)
                },
                'success': True,
                'message': f'successfully detected {shape}'
            }
            
            return result
            
        except Exception as e:
            return {
                'shape': {
                    'type': None,
                    'params': None,
                    'size_cm': None,
                    'size_cm_raw': None,
                    'size_pixels': None,
                    'detection_success': False,
                    'filtered_out': False,
                    'filter_reason': None
                },
                'statistics': {
                    'size_filtered': None,
                    'distance_filtered': D_corrected,
                    'filter_samples': len(self.avg_filter.size_history)
                },
                'success': False,
                'message': f'error: {str(e)}'
            }
    
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
            'distance_history': self.avg_filter.distance_history.copy()
        }