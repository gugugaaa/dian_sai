# result_stabilizer.py
from collections import deque

class ResultStabilizer:
    """
    接收算法的原始输出，应用规则进行稳定化处理。
    这是一个有状态的类，用于抑制抖动和做出更可靠的决策。
    """
    def __init__(self):
        # 在这里定义不同模式所需的状态变量
        # 例如，历史记录、计数器、当前稳定值等
        self.history = {
            'single': deque(maxlen=10),
            'overlap': deque(maxlen=20), # 为重叠模式保留更长的历史
            'digit': {} # 为每个数字方块维护一个状态
        }
        self.stable_results = {
            'single': None,
            'overlap': {'size_cm': None, 'count': 0},
            'digit': {}
        }
        print("结果稳定器初始化成功")

    def process(self, raw_result, mode):
        """
        主处理函数，根据模式分发到不同的处理逻辑。
        返回一个与原始结果格式相同的字典，但其中的值是稳定后的。
        """
        if not raw_result or not raw_result.get('success'):
            return raw_result

        if mode == 'single':
            return self._process_single(raw_result)
        elif mode == 'overlap':
            return self._process_overlap(raw_result)
        elif mode == 'digit':
            return self._process_digit(raw_result)
        
        return raw_result # 默认透传

    def _process_single(self, raw_result):
        # --- 在此实现 'single' 模式的稳定逻辑 ---
        # 当前算法内部已有移动平均滤波，这里先直接透传
        # 未来可以增加：如丢失目标后，保持显示最后N帧的结果
        shape_info = raw_result.get('shape', {})
        if shape_info.get('detection_success'):
            self.stable_results['single'] = raw_result
            return raw_result
        else:
            # 如果丢失目标，可以返回最后一次的稳定结果
            if self.stable_results['single']:
                 return self.stable_results['single']
            return raw_result


    def _process_overlap(self, raw_result):
        # --- 在此实现 'overlap' 模式的稳定逻辑 ---
        # 这是一个示例逻辑，你可以根据需求大幅修改
        
        # 提取当前帧的最小正方形尺寸
        current_size = None
        if raw_result.get('squares') and len(raw_result['squares']) > 0:
            # 假设第一个是最小的
            current_size = raw_result['squares'][0].get('size_cm_raw')

        # 规则1: 尺寸截断
        if current_size is not None and not (6.0 < current_size < 12.0):
            # 尺寸不合格，可以返回原始结果并附带一个警告，或者干脆不更新稳定结果
            raw_result['message'] += " | Stabilizer: Size out of range [6, 12]cm"
            return raw_result

        stable_size_info = self.stable_results['overlap']

        # 规则2: 稳定性更新
        if current_size is None:
            # 本帧未检测到，重置计数器
            stable_size_info['count'] = 0
        elif (stable_size_info['size_cm'] is None or 
              abs(current_size - stable_size_info['size_cm']) > 1.0): # 大跳变
            # 如果出现了一个和当前稳定值差异很大的新值，重置并将其设为候选
            stable_size_info['size_cm'] = current_size
            stable_size_info['count'] = 1
        else: # 小幅变化
            # 如果新值在稳定值附近，则平滑更新并增加计数
            alpha = 0.3 # 平滑因子
            stable_size_info['size_cm'] = (1 - alpha) * stable_size_info['size_cm'] + alpha * current_size
            stable_size_info['count'] += 1

        # 只有当一个值稳定出现超过N次时，才在结果中更新它
        # 这里只是一个简单的示例，你可以设计的更复杂
        if stable_size_info['count'] > 3 and raw_result.get('squares'):
            for sq in raw_result['squares']:
                # 将所有构造的正方形尺寸都更新为稳定值
                sq['size_cm'] = stable_size_info['size_cm']

        return raw_result


    def _process_digit(self, raw_result):
        # --- 在此实现 'digit' 模式的稳定逻辑 ---
        # 这个逻辑比较复杂，需要追踪每个方块
        # 此处仅作透传，留作未来的实现空间
        return raw_result

    def reset(self, mode=None):
        """重置状态"""
        if mode:
            if mode in self.history: self.history[mode].clear()
            if mode in self.stable_results: 
                # 根据不同模式的结构进行重置
                if mode == 'overlap':
                    self.stable_results[mode] = {'size_cm': None, 'count': 0}
                else:
                    self.stable_results[mode] = None
        else: # 重置所有
            self.history = {k: deque(maxlen=v.maxlen) if isinstance(v, deque) else {} for k, v in self.history.items()}
            self.stable_results = {
                'single': None,
                'overlap': {'size_cm': None, 'count': 0},
                'digit': {}
            }
        print(f"结果稳定器已重置 (模式: {mode or 'All'})")