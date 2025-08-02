# result_stabilizer.py
from collections import deque
import numpy as np

class ResultStabilizer:
    """
    接收算法的原始输出，应用规则进行稳定化处理。
    这是一个有状态的类，用于抑制抖动和做出更可靠的决策。
    """
    def __init__(self, digit_config=None):
        # 数字模式配置参数
        self.digit_config = digit_config or {
            'history_window': 10,           # 历史窗口大小
            'position_threshold': 50,       # 位置匹配阈值(像素)
            'jump_detection_window': 5,     # 跳变检测窗口
            'min_jump_count': 3,           # 最小跳变次数才认为是跳变
            'confidence_weight': 0.3,       # 置信度权重(用于选择稳定值)
            'stable_frames_required': 3,    # 需要连续稳定帧数才更新结果
        }
        
        # 在这里定义不同模式所需的状态变量
        self.history = {
            'single': deque(maxlen=10),
            'overlap': deque(maxlen=20), # 为重叠模式保留更长的历史
            'digit': {}  # 为每个数字方块维护一个状态 {square_id: SquareTracker}
        }
        self.stable_results = {
            'single': None,
            'overlap': {'size_cm': None, 'count': 0},
            'digit': {}
        }
        
        # 数字方块跟踪器的下一个ID
        self._next_square_id = 0
        
        print("结果稳定器初始化成功")
        print(f"数字模式配置: {self.digit_config}")

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
        """
        数字模式的稳定化处理 - 处理数字识别结果的跳变
        """
        squares = raw_result.get('squares', [])
        if not squares:
            return raw_result
        
        # 1. 匹配当前帧的方块与历史跟踪器
        matched_squares = self._match_squares_to_trackers(squares)
        
        # 2. 更新每个方块的跟踪器
        stabilized_squares = []
        for square, tracker_id in matched_squares:
            stabilized_square = self._update_square_tracker(square, tracker_id)
            stabilized_squares.append(stabilized_square)
        
        # 3. 清理过期的跟踪器
        self._cleanup_expired_trackers()
        
        # 4. 构建稳定化后的结果
        stabilized_result = raw_result.copy()
        stabilized_result['squares'] = stabilized_squares
        
        # 5. 添加稳定化统计信息
        stabilized_result['stabilizer_stats'] = {
            'active_trackers': len(self.history['digit']),
            'stabilized_count': sum(1 for sq in stabilized_squares if sq.get('stabilized', False))
        }
        
        return stabilized_result

    def _match_squares_to_trackers(self, squares):
        """
        将当前帧的方块匹配到现有的跟踪器
        返回: [(square, tracker_id), ...]
        """
        matched = []
        used_tracker_ids = set()
        
        for square in squares:
            center = np.array(square['center'])
            best_match = None
            min_distance = float('inf')
            
            # 寻找最近的跟踪器
            for tracker_id, tracker in self.history['digit'].items():
                if tracker_id in used_tracker_ids:
                    continue
                
                if tracker.get_recent_center() is not None:
                    distance = np.linalg.norm(center - tracker.get_recent_center())
                    if distance < self.digit_config['position_threshold'] and distance < min_distance:
                        min_distance = distance
                        best_match = tracker_id
            
            if best_match is not None:
                matched.append((square, best_match))
                used_tracker_ids.add(best_match)
            else:
                # 创建新的跟踪器
                new_tracker_id = self._create_new_tracker()
                matched.append((square, new_tracker_id))
        
        return matched

    def _create_new_tracker(self):
        """创建新的方块跟踪器"""
        tracker_id = f"tracker_{self._next_square_id}"
        self._next_square_id += 1
        
        self.history['digit'][tracker_id] = SquareTracker(
            tracker_id, 
            self.digit_config
        )
        
        return tracker_id

    def _update_square_tracker(self, square, tracker_id):
        """更新指定跟踪器并返回稳定化后的方块信息"""
        tracker = self.history['digit'][tracker_id]
        
        # 更新跟踪器
        tracker.update(square)
        
        # 获取稳定化后的数字
        stabilized_digit, is_stabilized = tracker.get_stable_digit()
        
        # 创建稳定化后的方块副本
        stabilized_square = square.copy()
        
        if is_stabilized and stabilized_digit is not None:
            stabilized_square['digit'] = stabilized_digit
            stabilized_square['confidence'] = tracker.get_stable_confidence()
            stabilized_square['stabilized'] = True
            stabilized_square['stabilizer_info'] = {
                'tracker_id': tracker_id,
                'jump_detected': tracker.has_jump_pattern(),
                'history_size': len(tracker.digit_history),
                'unique_digits': tracker.get_unique_digits()
            }
        else:
            stabilized_square['stabilized'] = False
            stabilized_square['stabilizer_info'] = {
                'tracker_id': tracker_id,
                'jump_detected': tracker.has_jump_pattern(),
                'history_size': len(tracker.digit_history),
                'reason': 'insufficient_history' if len(tracker.digit_history) < self.digit_config['stable_frames_required'] else 'no_recognition'
            }
        
        return stabilized_square

    def _cleanup_expired_trackers(self, max_idle_frames=20):
        """清理长时间未更新的跟踪器"""
        to_remove = []
        for tracker_id, tracker in self.history['digit'].items():
            if tracker.frames_since_update > max_idle_frames:
                to_remove.append(tracker_id)
        
        for tracker_id in to_remove:
            del self.history['digit'][tracker_id]

    def reset(self, mode=None):
        """重置状态"""
        if mode:
            if mode in self.history: 
                if mode == 'digit':
                    self.history[mode] = {}
                else:
                    self.history[mode].clear()
            if mode in self.stable_results: 
                # 根据不同模式的结构进行重置
                if mode == 'overlap':
                    self.stable_results[mode] = {'size_cm': None, 'count': 0}
                else:
                    self.stable_results[mode] = None
        else: # 重置所有
            self.history = {
                'single': deque(maxlen=10),
                'overlap': deque(maxlen=20),
                'digit': {}
            }
            self.stable_results = {
                'single': None,
                'overlap': {'size_cm': None, 'count': 0},
                'digit': {}
            }
            self._next_square_id = 0
        
        print(f"结果稳定器已重置 (模式: {mode or 'All'})")

    def get_digit_statistics(self):
        """获取数字模式的统计信息"""
        stats = {
            'active_trackers': len(self.history['digit']),
            'trackers': {}
        }
        
        for tracker_id, tracker in self.history['digit'].items():
            stats['trackers'][tracker_id] = {
                'history_size': len(tracker.digit_history),
                'unique_digits': tracker.get_unique_digits(),
                'has_jump': tracker.has_jump_pattern(),
                'stable_digit': tracker.get_stable_digit()[0],
                'frames_since_update': tracker.frames_since_update
            }
        
        return stats


class SquareTracker:
    """单个数字方块的跟踪器"""
    
    def __init__(self, tracker_id, config):
        self.tracker_id = tracker_id
        self.config = config
        
        # 历史记录
        self.digit_history = deque(maxlen=config['history_window'])
        self.position_history = deque(maxlen=config['history_window'])
        self.confidence_history = deque(maxlen=config['history_window'])
        
        # 状态
        self.frames_since_update = 0
        self.stable_digit = None
        self.stable_confidence = 0.0
        self.last_update_frame = 0
        
    def update(self, square):
        """更新跟踪器状态"""
        self.frames_since_update = 0
        
        # 记录位置
        center = np.array(square['center'])
        self.position_history.append(center)
        
        # 记录数字识别结果
        if square.get('recognition_success', False):
            digit = square['digit']
            confidence = square['confidence']
            
            self.digit_history.append({
                'digit': digit,
                'confidence': confidence,
                'frame': self.last_update_frame
            })
            self.confidence_history.append(confidence)
        
        self.last_update_frame += 1
        
        # 更新稳定数字
        self._update_stable_digit()
    
    def _update_stable_digit(self):
        """更新稳定的数字值"""
        if len(self.digit_history) < self.config['stable_frames_required']:
            return
        
        # 检查是否存在跳变模式
        if self.has_jump_pattern():
            # 存在跳变，选择最小的非0数字
            non_zero_digits = [entry['digit'] for entry in self.digit_history 
                             if entry['digit'] != 0]
            
            if non_zero_digits:
                # 选择最小的非0数字
                min_digit = min(non_zero_digits)
                
                # 计算该数字的平均置信度
                confidences = [entry['confidence'] for entry in self.digit_history 
                             if entry['digit'] == min_digit]
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                self.stable_digit = min_digit
                self.stable_confidence = avg_confidence
            else:
                # 如果没有非0数字，选择最常见的数字
                self._select_most_frequent_digit()
        else:
            # 没有跳变，选择最常见的数字
            self._select_most_frequent_digit()
    
    def _select_most_frequent_digit(self):
        """选择最频繁出现的数字作为稳定值"""
        if not self.digit_history:
            return
        
        # 统计数字出现频率
        digit_counts = {}
        digit_confidences = {}
        
        for entry in self.digit_history:
            digit = entry['digit']
            confidence = entry['confidence']
            
            if digit not in digit_counts:
                digit_counts[digit] = 0
                digit_confidences[digit] = []
            
            digit_counts[digit] += 1
            digit_confidences[digit].append(confidence)
        
        # 选择最频繁的数字，如果频率相同则选择置信度更高的
        best_digit = None
        best_score = -1
        
        for digit, count in digit_counts.items():
            avg_confidence = np.mean(digit_confidences[digit])
            # 综合考虑频率和置信度
            score = count + self.config['confidence_weight'] * avg_confidence
            
            if score > best_score:
                best_score = score
                best_digit = digit
        
        if best_digit is not None:
            self.stable_digit = best_digit
            self.stable_confidence = np.mean(digit_confidences[best_digit])
    
    def has_jump_pattern(self):
        """检测是否存在跳变模式"""
        if len(self.digit_history) < self.config['jump_detection_window']:
            return False
        
        # 检查最近N帧中的数字变化
        recent_digits = [entry['digit'] for entry in 
                        list(self.digit_history)[-self.config['jump_detection_window']:]]
        
        unique_digits = set(recent_digits)
        
        # 如果在检测窗口内出现了多个不同数字，认为是跳变
        return len(unique_digits) >= self.config['min_jump_count']
    
    def get_stable_digit(self):
        """获取稳定的数字值"""
        is_stable = (len(self.digit_history) >= self.config['stable_frames_required'] 
                    and self.stable_digit is not None)
        return self.stable_digit, is_stable
    
    def get_stable_confidence(self):
        """获取稳定数字的置信度"""
        return self.stable_confidence
    
    def get_recent_center(self):
        """获取最近的中心位置"""
        if self.position_history:
            return self.position_history[-1]
        return None
    
    def get_unique_digits(self):
        """获取历史中出现过的所有不同数字"""
        return list(set(entry['digit'] for entry in self.digit_history))
    
    def increment_idle_frames(self):
        """增加空闲帧计数"""
        self.frames_since_update += 1