"""
Face preprocessing utilities for person tracking system.
Handles face extraction and alignment from high-resolution frames.
"""

import cv2
import numpy as np
import time
import os
from skimage import transform as trans


def quick_face_extract(frame_resized, bbox_resized, frame_original, scale_x, scale_y, yunet_detector, min_confidence=0.85, debug_count=0):
    """
    🔥 FIXED VERSION: Align face 112x112 theo chuẩn InsightFace (giống align_face_224)
    
    Args:
        frame_resized: Frame 640x480 (chỉ dùng để reference)
        bbox_resized: Bbox trên frame 640x480 [x1, y1, x2, y2]
        frame_original: Frame 2K gốc từ camera (2560x1920)
        scale_x: Tỷ lệ width (2560/640 = 4.0)
        scale_y: Tỷ lệ height (1920/480 = 4.0)
        yunet_detector: YuNet face detector
        min_confidence: Ngưỡng confidence tối thiểu
        debug_count: Frame count để debug output
    """
    try:
        # 🔥 BƯỚC 1: Scale bbox từ 640x480 LÊN 2K
        x1_640, y1_640, x2_640, y2_640 = map(int, bbox_resized)
        
        x1_2k = int(x1_640 * scale_x)
        y1_2k = int(y1_640 * scale_y)
        x2_2k = int(x2_640 * scale_x)
        y2_2k = int(y2_640 * scale_y)
        
        # Clamp về giới hạn frame 2K
        h_2k, w_2k = frame_original.shape[:2]
        x1_2k = max(0, min(x1_2k, w_2k))
        y1_2k = max(0, min(y1_2k, h_2k))
        x2_2k = max(0, min(x2_2k, w_2k))
        y2_2k = max(0, min(y2_2k, h_2k))
        
        if x2_2k <= x1_2k or y2_2k <= y1_2k:
            return None, 0
        
        # 🔥 BƯỚC 2: Padding trên frame 2K
        pad_w = int((x2_2k - x1_2k) * 0.2)
        pad_h = int((y2_2k - y1_2k) * 0.2)
        px1 = max(0, x1_2k - pad_w)
        py1 = max(0, y1_2k - pad_h)
        px2 = min(w_2k, x2_2k + pad_w)
        py2 = min(h_2k, y2_2k + pad_h)
        
        # 🔥 BƯỚC 3: Cắt person crop từ FRAME 2K GỐC
        person_crop = frame_original[py1:py2, px1:px2]
        if person_crop.size == 0:
            return None, 0
        
        # 🔥 BƯỚC 4: YuNet detect trên person crop 2K (chất lượng cao)
        dets, lms = yunet_detector.detect(person_crop, threshold=min_confidence)
        if dets is None or len(dets) == 0:
            return None, 0
        
        # Lấy face có confidence cao nhất
        best_idx = np.argmax(dets[:, 4])
        face_conf = dets[best_idx, 4]
        
        if face_conf < min_confidence:
            return None, 0
        
        face_landmarks = lms[best_idx].reshape(5, 2)
        
        # ✅ BƯỚC 5: ALIGN TRỰC TIẾP theo chuẩn InsightFace (giống align_face_224)
        # Tọa độ landmarks CHUẨN cho 112x112 (từ InsightFace/ArcFace)
        dst_points = np.array([
            [38.2946, 51.6963],    # right eye
            [73.5318, 51.5014],    # left eye
            [56.0252, 71.7366],    # nose
            [41.5493, 92.3655],    # right mouth corner
            [70.7299, 92.2041]     # left mouth corner
        ], dtype=np.float32)
        
        src_points = face_landmarks.astype(np.float32)
        
        # SimilarityTransform tự động tính toán rotation + scale + translation
        tform = trans.SimilarityTransform()
        tform.estimate(src_points, dst_points)
        M = tform.params[0:2, :]
        
        # Warp ảnh TRỰC TIẾP từ person_crop (không cần crop face_only)
        face_aligned = cv2.warpAffine(
            person_crop, M, (112, 112),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LINEAR
        )
        
        face_final = face_aligned
        
        # 🔥 DEBUG: Lưu face crop từ 2K (mỗi 30 frames)
        try: 
            if face_final.size > 0:
                timestamp = int(time.time() * 1000)
                
                # Lưu vào folder cũ (backward compatible)
                DEBUG_FACES_DIR = "debug_faces"
                if not os.path.exists(DEBUG_FACES_DIR):
                    os.makedirs(DEBUG_FACES_DIR)
                    
                filename = f"face_{timestamp}_conf{face_conf:.2f}.jpg"
                debug_path = os.path.join(DEBUG_FACES_DIR, filename)
                cv2.imwrite(debug_path, face_final)
                
                # 🔥 Lưu vào folder mới (mỗi 30 frames) để đánh dấu từ 2K
                if debug_count % 30 == 0 and debug_count > 0:
                    debug_2k_dir = "debug_faces_2k"
                    if not os.path.exists(debug_2k_dir):
                        os.makedirs(debug_2k_dir)
                    
                    filename_2k = f"face_{timestamp}_from2K_conf{face_conf:.2f}.jpg"
                    debug_path_2k = os.path.join(debug_2k_dir, filename_2k)
                    success = cv2.imwrite(debug_path_2k, face_final)
                    if success:
                        print(f"🖼️ [DEBUG] Saved face from 2K: {debug_path_2k}")

        except Exception as save_err:
            print(f"⚠️ [DEBUG SAVE ERROR]: {save_err}")
            
        q_score = face_conf
        return face_final, q_score
        
    except Exception as e:
        # In lỗi tổng quát để biết tại sao hàm chết
        print(f"⚠️ Extract error details: {e}")
        import traceback
        traceback.print_exc() # In chi tiết dòng lỗi
        return None, 0
