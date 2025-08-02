# drawing_utils.py
import cv2
import numpy as np

class DrawingUtils:
    """负责将检测和稳定后的结果绘制到图像上"""
    
    TARGET_WIDTH = 300  # 目标宽度像素值
    
    def __init__(self):
        self.colors = {
            'border': (255, 0, 0),
            'shape': (0, 255, 0),
            'text': (0, 255, 255),
            'polygon': (0, 165, 255),
            'vertex': (0, 0, 255),
            'square_constructed': (255, 0, 255),
            'digit_box': (255, 255, 0)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _resize_frame_proportionally(self, frame):
        """等比例调整图像大小到目标宽度"""
        height, width = frame.shape[:2]
        if width == 0:
            return frame
        
        scale_factor = self.TARGET_WIDTH / width
        new_height = int(height * scale_factor)
        resized_frame = cv2.resize(frame, (self.TARGET_WIDTH, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame, scale_factor

    def draw(self, frame, stabilized_result, mode):
        """主绘制函数，根据模式调用不同的子函数"""
        if not stabilized_result or not stabilized_result.get('success', False):
            resized_frame, _ = self._resize_frame_proportionally(frame)
            return resized_frame

        # 先调整图像大小
        resized_frame, scale_factor = self._resize_frame_proportionally(frame)

        if mode == 'single':
            return self._draw_single_shape(resized_frame, stabilized_result, scale_factor)
        elif mode == 'overlap':
            return self._draw_overlap_square(resized_frame, stabilized_result, scale_factor)
        elif mode == 'digit':
            return self._draw_digits(resized_frame, stabilized_result, scale_factor)
        return resized_frame

    def _draw_single_shape(self, frame, result, scale_factor):
        shape_info = result.get('shape', {})
        if shape_info.get('detection_success'):
            shape_type = shape_info['type']
            cv2.putText(frame, f"Shape: {shape_type}", (10, 30), self.font, 0.7, self.colors['text'], 2)
            
            size_cm = shape_info.get('size_cm')
            if size_cm is not None:
                cv2.putText(frame, f"Size: {size_cm:.2f} cm", (10, 60), self.font, 0.7, self.colors['text'], 2)
        
        message = result.get('message', 'No shape detected')
        cv2.putText(frame, message, (10, frame.shape[0] - 10), self.font, 0.5, self.colors['text'], 1)
        return frame

    def _draw_overlap_square(self, frame, result, scale_factor):
        # 绘制多边形轮廓
        polygons = result.get('polygons', {})
        if polygons.get('outer_polygons'):
            for poly in polygons['outer_polygons']:
                scaled_poly = (poly['approx'] * scale_factor).astype(np.int32)
                cv2.drawContours(frame, [scaled_poly], -1, self.colors['polygon'], 2)
        
        # 绘制凸90度角顶点
        construction_info = result.get('construction_info', {})
        if construction_info.get('convex_90_vertices'):
            for v_info in construction_info['convex_90_vertices']:
                scaled_coords = (int(v_info['coordinates'][0] * scale_factor), 
                               int(v_info['coordinates'][1] * scale_factor))
                cv2.circle(frame, scaled_coords, 5, self.colors['vertex'], -1)

        # 绘制构造出的正方形
        if result.get('squares'):
            for square in result['squares']:
                scaled_corners = (square['corners'] * scale_factor).astype(np.int32)
                cv2.polylines(frame, [scaled_corners], True, self.colors['square_constructed'], 3)
                # 绘制尺寸
                size_cm = square.get('size_cm')
                if size_cm:
                    # 计算正方形左上角位置作为文字起始位置
                    min_x = np.min(scaled_corners[:, 0])
                    min_y = np.min(scaled_corners[:, 1])
                    text_pos = (min_x + 5, min_y + 20)  # 左上角内部偏移一点
                    cv2.putText(frame, f"Side: {size_cm:.2f} cm", text_pos, self.font, 0.4, self.colors['square_constructed'], 1)

        message = result.get('message', 'Processing...')
        cv2.putText(frame, message, (10, frame.shape[0] - 10), self.font, 0.5, self.colors['text'], 1)
        return frame

    def _draw_digits(self, frame, result, scale_factor):
        squares = result.get('squares', [])
        for sq in squares:
            if sq.get('filtered_out'): continue

            # 绘制正方形边框
            scaled_corners = (np.array(sq['corners']) * scale_factor).astype(np.int32)
            cv2.polylines(frame, [scaled_corners], True, self.colors['digit_box'], 2)
            
            # 计算正方形左上角位置作为文字起始位置
            min_x = np.min(scaled_corners[:, 0])
            min_y = np.min(scaled_corners[:, 1])
            text_pos = (min_x + 5, min_y + 20)  # 左上角内部偏移一点
            
            size_cm = sq.get('size_cm')
            if sq['recognition_success']:
                digit = sq['digit']
                confidence = sq['confidence']
                text = f"Digit: {digit} ({confidence:.2f})"
                cv2.putText(frame, text, text_pos, self.font, 0.4, self.colors['shape'], 1)
                if size_cm:
                     cv2.putText(frame, f"{size_cm:.1f}cm", (text_pos[0], text_pos[1] + 15), self.font, 0.4, self.colors['text'], 1)
            elif size_cm:
                 cv2.putText(frame, f"Size: {size_cm:.1f}cm", text_pos, self.font, 0.4, self.colors['text'], 1)

        message = result.get('message', 'Processing...')
        cv2.putText(frame, message, (10, frame.shape[0] - 10), self.font, 0.5, self.colors['text'], 1)
        return frame