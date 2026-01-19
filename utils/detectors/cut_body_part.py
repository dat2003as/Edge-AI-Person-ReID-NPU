# utils/cut_body_part.py
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Các kết nối giữa các điểm keypoint của MediaPipe để vẽ skeleton
MEDIAPIPE_EDGES = [
    # Khuôn mặt
    (0, 1),   # Mũi -> Mắt trong trái
    (1, 2),   # Mắt trong trái -> Mắt trái
    (2, 3),   # Mắt trái -> Mắt ngoài trái
    (3, 7),   # Mắt ngoài trái -> Tai trái
    (0, 4),   # Mũi -> Mắt trong phải
    (4, 5),   # Mắt trong phải -> Mắt phải
    (5, 6),   # Mắt phải -> Mắt ngoài phải
    (6, 8),   # Mắt ngoài phải -> Tai phải
    (9, 10),  # Miệng trái -> Miệng phải

    # Thân trên
    (11, 12), # Vai trái -> Vai phải
    (11, 23), # Vai trái -> Hông trái
    (12, 24), # Vai phải -> Hông phải
    (23, 24), # Hông trái -> Hông phải

    # Tay trái
    (11, 13), # Vai trái -> Khuỷu tay trái
    (13, 15), # Khuỷu tay trái -> Cổ tay trái
    (15, 17), # Cổ tay trái -> Ngón út trái (pinky)
    (15, 19), # Cổ tay trái -> Ngón trỏ trái (index)
    (15, 21), # Cổ tay trái -> Ngón cái trái
    (17, 19), # Ngón út trái -> Ngón trỏ trái

    # Tay phải
    (12, 14), # Vai phải -> Khuỷu tay phải
    (14, 16), # Khuỷu tay phải -> Cổ tay phải
    (16, 18), # Cổ tay phải -> Ngón út phải (pinky)
    (16, 20), # Cổ tay phải -> Ngón trỏ phải (index)
    (16, 22), # Cổ tay phải -> Ngón cái phải
    (18, 20), # Ngón út phải -> Ngón trỏ phải

    # Chân trái
    (23, 25), # Hông trái -> Đầu gối trái
    (25, 27), # Đầu gối trái -> Mắt cá trái
    (27, 29), # Mắt cá trái -> Gót chân trái
    (27, 31), # Mắt cá trái -> Ngón chân cái trái
    (29, 31), # Gót chân trái -> Ngón chân cái trái

    # Chân phải
    (24, 26), # Hông phải -> Đầu gối phải
    (26, 28), # Đầu gối phải -> Mắt cá phải
    (28, 30), # Mắt cá phải -> Gót chân phải
    (28, 32), # Mắt cá phải -> Ngón chân cái phải
    (30, 32)  # Gót chân phải -> Ngón chân cái phải
]

def compute_head_bbox(person_keypoints: np.ndarray, scale=1, adjust_ratio=0.35) -> Optional[Tuple[int, int, int, int]]:
    """
    Tính toán vùng giới hạn (bounding box) cho vùng đầu dựa trên keypoints.
    Sử dụng tâm và bán kính lớn nhất để tạo vùng cắt hình vuông.
    
    Args:
        person_keypoints (np.ndarray): Mảng keypoints cho vùng đầu, shape (N, 3).
        scale (float): Hệ số mở rộng vùng cắt.
        adjust_ratio (float): Tỷ lệ điều chỉnh center_y hướng lên trên.
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) hoặc None.
    """
    # Lọc các keypoint có confidence > 0
    keypoints = np.array([kp for kp in person_keypoints if kp[2] > 0.1]) # <<< FIX: Lọc bằng confidence

    if keypoints.shape[0] < 2:  # Cần ít nhất 2 keypoint hợp lệ để tính toán
        return None

    # Chỉ tính toán trên tọa độ (x, y)
    xy_keypoints = keypoints[:, :2] # <<< FIX: Lấy riêng tọa độ (x,y)

    # Tính tâm trung bình
    center_x, center_y = np.mean(xy_keypoints, axis=0)

    # Tính bán kính lớn nhất
    # <<< FIX: Thực hiện phép trừ trên mảng (N, 2)
    distances = np.linalg.norm(xy_keypoints - [center_x, center_y], axis=1)
    radius = np.max(distances) * scale

    # Điều chỉnh center_y hướng lên trên
    center_y -= center_y * adjust_ratio  

    # Xác định tọa độ vùng cắt hình vuông
    x_min = int(center_x - radius)
    x_max = int(center_x + radius)
    y_min = int(center_y - radius)
    y_max = int(center_y + radius)

    return x_min, y_min, x_max, y_max

def extract_head_from_frame(frame: np.ndarray, person_keypoints: np.ndarray, scale=1) -> Optional[Tuple[int, int, int, int]]:
    """
    Trích xuất bounding box vùng đầu từ frame dựa trên keypoints.
    """
    try:
        # Lấy keypoints cho vùng đầu (Mũi, Mắt, Tai, Vai)
        head_indices = [0, 1, 2, 3, 4, 5, 6]
        # Kiểm tra xem có đủ keypoints không
        if person_keypoints.shape[0] <= max(head_indices):
             logger.error("Mảng keypoints không đủ phần tử để trích xuất đầu.")
             return None
        
        keypoints_for_head = person_keypoints[head_indices]
        
        bbox = compute_head_bbox(keypoints_for_head, scale)
        if bbox is None:
            logger.warning("Không đủ keypoints hợp lệ để tính bbox đầu.")
            return None

        x_min, y_min, x_max, y_max = bbox

        # Giới hạn trong kích thước frame
        h, w = frame.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            #logger.warning("Bounding box đầu không hợp lệ sau khi cắt theo frame.")
            return None
        
        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        logger.error(f"Lỗi trong extract_head_from_frame: {e}", exc_info=True)
        return None


def compute_arm_region(frame: np.ndarray, start_point: np.ndarray, end_point: np.ndarray, thickness=100) -> np.ndarray:
    """
    Tính toán mask vùng cánh tay.
    """
    # <<< FIX: Chỉ lấy tọa độ (x,y) để tính toán và vẽ
    start_point_xy = start_point[:2]
    end_point_xy = end_point[:2]

    start_point_int = tuple(map(int, start_point_xy))
    end_point_int = tuple(map(int, end_point_xy))
    
    line_length = np.linalg.norm(start_point_xy - end_point_xy)
    
    frame_height = frame.shape[0]
    new_thickness = max(1, int(thickness * (frame_height / (line_length + 1e-6))))
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.line(mask, start_point_int, end_point_int, 255, new_thickness)
    
    return mask

def extract_arm_region(frame: np.ndarray, start_point: np.ndarray, end_point: np.ndarray, thickness=25, conf_threshold=0.3) -> Optional[Tuple[int, int, int, int]]:
    """
    Tính toán bounding box vùng cánh tay.
    """
    try:
        # <<< FIX: Kiểm tra confidence thay vì tọa độ (0,0)
        if start_point[2] < conf_threshold or end_point[2] < conf_threshold:
            return None

        mask = compute_arm_region(frame, start_point, end_point, thickness)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        x, y, w, h = cv2.boundingRect(contours[0])

        if w == 0 or h == 0:
            return None

        return (x, y, x + w, y + h)
    except Exception as e:
        logger.error(f"Lỗi trong extract_arm_region: {e}", exc_info=True)
        return None

def extract_body_parts_from_frame(frame: np.ndarray, person_keypoints: np.ndarray) -> Dict:
    """
    Cắt các vùng cơ thể từ frame dựa trên person_keypoints.
    Trả về một dictionary chứa bounding box của các vùng.
    """
    body_parts = {}
    
    # Cắt vùng đầu
    body_parts['head'] = extract_head_from_frame(frame, person_keypoints) 

    # Cắt vùng cẳng tay phải (khuỷu tay -> cổ tay)
    if person_keypoints.shape[0] > 10: # Đảm bảo có đủ keypoints
        right_elbow = person_keypoints[8]
        right_wrist = person_keypoints[10]
        body_parts['right_forearm'] = extract_arm_region(frame, right_elbow, right_wrist)

        # Cắt vùng cẳng tay trái
        left_elbow = person_keypoints[7]
        left_wrist = person_keypoints[9]
        body_parts['left_forearm'] = extract_arm_region(frame, left_elbow, left_wrist)
    else:
        body_parts['right_forearm'] = None
        body_parts['left_forearm'] = None


    return body_parts