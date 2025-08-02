import numpy as np

class MovingAverageFilter:
    """移动平均值滤波器，用于减少测量结果的跳动"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.size_history = []
        self.distance_history = []
    
    def update(self, size, distance):
        """更新历史数据并返回平均值"""
        # 更新尺寸历史
        self.size_history.append(size)
        if len(self.size_history) > self.window_size:
            self.size_history.pop(0)
        
        # 更新距离历史
        self.distance_history.append(distance)
        if len(self.distance_history) > self.window_size:
            self.distance_history.pop(0)
        
        # 返回平均值
        avg_size = sum(self.size_history) / len(self.size_history)
        avg_distance = sum(self.distance_history) / len(self.distance_history)
        
        return avg_size, avg_distance
    
    def reset(self):
        """重置历史数据"""
        self.size_history.clear()
        self.distance_history.clear()
    
    def get_statistics(self):
        """获取滤波器统计信息"""
        return {
            'filter_samples': len(self.size_history),
            'size_history': self.size_history.copy(),
            'distance_history': self.distance_history.copy()
        }