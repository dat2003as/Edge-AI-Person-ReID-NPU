import cv2
import numpy as np
from ultralytics import YOLO
from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)
import os
import time
from typing import Optional
# Import FaceDetection - cần đảm bảo path đúng
from utils.detectors.mdp_face_detection_task import FaceDetection 
# Import extract_head_from_frame để cắt headbox
from utils.detectors.cut_body_part import extract_head_from_frame

### <<< CẤU HÌNH NGƯỠNG VÀ TỈ LỆ KẾT HỢP >>> ###
# NGƯỠNG ĐỂ QUYẾT ĐỊNH SỬ DỤNG MODEL NÀO 
FACE_CONFIDENCE_HIGH_THRESHOLD = 0.8    # Ngưỡng cao: chỉ dùng face model
FACE_CONFIDENCE_LOW_THRESHOLD = 0.6     # Ngưỡng thấp: dưới đây chỉ dùng pose model

# TỈ LỆ KẾT HỢP KHI FACE CONFIDENCE TRONG KHOẢNG 0.5-0.7
FACE_WEIGHT = 0.6   # Trọng số cho face model (70%)
POSE_WEIGHT = 1 - FACE_WEIGHT     # Trọng số cho pose model (30%)

# CONFIDENCE MẶC ĐỊNH KHI CHỈ SỬ DỤNG POSE MODEL
DEFAULT_POSE_CONFIDENCE = 0.8
MIN_PERSON_AREA = 150 * 100 # Bỏ qua các box có diện tích nhỏ hơn 100x100 pixels
MAX_ASPECT_RATIO_FOR_POSE = 5 # Tỉ lệ H/W tối đa, tránh các vật thể quá dài, gầy
MIN_ASPECT_RATIO_FOR_POSE = 1.5 # Tỉ lệ H/W tối thiểu, để BỎ QUA các box gần vuông

# NGƯỠNG TỐI THIỂU CHO POSE MODEL KHI COMBINE
MIN_POSE_CONFIDENCE_FOR_COMBINE = 0.65  # Ngưỡng tối thiểu khi combine

# SỐ LƯỢNG KEYPOINTS TỐI THIỂU CẦN THIẾT CHO POSE MODEL
MIN_KEYPOINTS_FOR_POSE = 27  # Cần ít nhất 10 keypoints có confidence > 0.3

CLASS_NAME = ['female', 'male']

class GenderClassification:
    def __init__(self, gender_face_model_path, gender_pose_model_path, device='cpu'):
        """
        Khởi tạo hệ thống nhận diện giới tính
        Args:
            gender_face_model_path: Đường dẫn model face
            gender_pose_model_path: Đường dẫn model pose
            device: 'cpu' hoặc 'cuda'
        """
        self.device = device
        
        # Tải các model
        self.face_detection = FaceDetection(min_detection_confidence=0.5)
        self.gender_face_model = YOLO(gender_face_model_path)
        self.gender_pose_model = YOLO(gender_pose_model_path)
        self.face_classes = CLASS_NAME
        self.pose_classes = CLASS_NAME

        self.save_dir = None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"Ảnh nhận diện giới tính sẽ được lưu tại: {self.save_dir}")

        
        print("GenderClassification initialized successfully")


    def predict(self, human_img, keypoints=None):
        """
        Hàm chính để nhận diện giới tính từ ảnh người đã crop
        
        Returns:
            Dictionary với format mới:
            {
                "gender": "male"/"female"/"unknown",
                "confidence": float (0-1),
                "strategy": "face_only"/"combined"/"pose_only"/"unknown"
            }
        """
        # Kiểm tra input
        if human_img is None or human_img.size == 0:
            return {"gender": "unknown", "confidence": 0.0, "strategy": "unknown"}
        
        # Tạo pseudo bbox từ kích thước ảnh
        h, w = human_img.shape[:2]
        pseudo_bbox = np.array([0, 0, w, h])
        
        # Gọi hàm classify nội bộ với keypoints
        result = self._classify_internal(human_img, pseudo_bbox, keypoints)
        
        # Chuyển đổi format output
        gender = result.get("gender", "Unknown").lower()
        if gender == "unknown":
            gender = "unknown" # Đảm bảo tên nhất quán

        ### <<< THAY ĐỔI: Thêm chức năng lưu ảnh và trả về strategy >>> ###
        # Lưu ảnh nếu thư mục đã được cấu hình
        if self.save_dir:
            self._save_gender_image(human_img, gender)

        logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info(f"Gender: {gender}, Confidence: {result.get('confidence', 0.0):.2f}, Strategy: {result.get('strategy', 'unknown')}")
        
        # Trả về kết quả đầy đủ
        return {
            "gender": gender,
            "confidence": result.get("confidence", 0.0),
            "strategy": result.get("strategy", "unknown")
        }

    ### <<< HÀM MỚI: Lưu ảnh theo giới tính để finetune >>> ###
    def _save_gender_image(self, image: np.ndarray, gender: str):
        """
        Lưu ảnh của người vào thư mục con tương ứng với giới tính.
        
        Args:
            image (np.ndarray): Ảnh bbox của người.
            gender (str): Giới tính đã được dự đoán ('male', 'female', 'unknown').
        """
        if not self.save_dir:
            return

        try:
            # Đảm bảo gender là một trong các giá trị hợp lệ
            gender_folder = gender if gender in ['male', 'female'] else 'unknown'
            
            target_dir = os.path.join(self.save_dir, gender_folder)
            os.makedirs(target_dir, exist_ok=True)
            
            # Tạo tên file duy nhất bằng timestamp
            filename = f"{int(time.time() * 1000)}.jpg"
            filepath = os.path.join(target_dir, filename)
            
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            logger.info(f"Đã lưu ảnh giới tính vào: {filepath}")

        except Exception as e:
            logger.error(f"Không thể lưu ảnh giới tính: {e}", exc_info=True)
    def _classify_internal(self, person_roi, person_bbox, keypoints=None):
        """
        Logic nhận diện nội bộ
        """
        # Phát hiện khuôn mặt với preprocessing headbox
        face_bbox, face_confidence = self._detect_best_face(person_roi, keypoints)
        logger.info(f"face_bbox: {face_bbox}, face_confidence: {face_confidence}")
        # Xác định chiến lược
        if face_confidence >= FACE_CONFIDENCE_HIGH_THRESHOLD:
            strategy = "face_only"
        elif face_confidence >= FACE_CONFIDENCE_LOW_THRESHOLD:
            strategy = "combined"
        else:
            strategy = "pose_only"
        logger.info(f"strategy: {strategy}")
        result = {"gender": "Unknown", "confidence": 0.0, "strategy": strategy}
        
        # Thực thi chiến lược
        if strategy == "face_only":
            if face_bbox is not None:
                x1_f, y1_f, x2_f, y2_f = map(int, face_bbox)
                face_roi = person_roi[y1_f:y2_f, x1_f:x2_f]
                if face_roi.size > 0:
                    face_probs = self._predict_face(face_roi)
                    pred_idx = np.argmax(face_probs)
                    result["gender"] = self.face_classes[pred_idx]
                    result["confidence"] = float(face_probs[pred_idx])
        
        elif strategy == "combined":
            if not self._is_pose_reliable(person_bbox, keypoints):
                # Fallback to face only
                if face_bbox is not None:
                    x1_f, y1_f, x2_f, y2_f = map(int, face_bbox)
                    face_roi = person_roi[y1_f:y2_f, x1_f:x2_f]
                    if face_roi.size > 0:
                        face_probs = self._predict_face(face_roi)
                        pred_idx = np.argmax(face_probs)
                        result["gender"] = self.face_classes[pred_idx]
                        result["confidence"] = float(face_probs[pred_idx])
            else:
                # Kết hợp face và pose
                face_probs = np.zeros(len(self.face_classes))
                if face_bbox is not None:
                    x1_f, y1_f, x2_f, y2_f = map(int, face_bbox)
                    face_roi = person_roi[y1_f:y2_f, x1_f:x2_f]
                    if face_roi.size > 0:
                        face_probs = self._predict_face(face_roi)
                
                pose_probs = self._predict_pose(person_roi)
                
                # Kiểm tra ngưỡng tối thiểu cho pose model khi combine
                pose_max_confidence = float(np.max(pose_probs))
                if pose_max_confidence < MIN_POSE_CONFIDENCE_FOR_COMBINE:
                    # Nếu pose confidence quá thấp, chỉ dùng face
                    if np.max(face_probs) > 0:
                        pred_idx = np.argmax(face_probs)
                        result["gender"] = self.face_classes[pred_idx]
                        result["confidence"] = float(face_probs[pred_idx])
                else:
                    # Combine bình thường
                    combined_probs = (face_probs * FACE_WEIGHT) + (pose_probs * POSE_WEIGHT)
                    pred_idx = np.argmax(combined_probs)
                    result["gender"] = self.face_classes[pred_idx]
                    result["confidence"] = float(combined_probs[pred_idx])
        
        elif strategy == "pose_only":
            logger.info(f"pose reiliable {self._is_pose_reliable(person_bbox, keypoints)}")
            if self._is_pose_reliable(person_bbox, keypoints):
                pose_probs = self._predict_pose(person_roi)
                pred_idx = np.argmax(pose_probs)
                confidence = float(pose_probs[pred_idx])
                
                if confidence >= DEFAULT_POSE_CONFIDENCE:
                    result["gender"] = self.pose_classes[pred_idx]
                    result["confidence"] = confidence
        logger.info(f"Strategy: {strategy}, Result: {result}")
        return result
    
    def _detect_best_face(self, person_roi, keypoints=None, top_margin=0.1):
        """Phát hiện khuôn mặt tốt nhất với preprocessing headbox nếu có keypoints"""
        try:
            # Cắt headbox trước nếu có keypoints
            roi_to_detect = person_roi
            head_bbox = None
            
            if keypoints is not None and keypoints.shape[0] >= 7:  # Cần ít nhất 7 keypoints cho đầu
                # Extract headbox từ keypoints
                head_bbox = extract_head_from_frame(person_roi, keypoints, scale=1.2)
                
                if head_bbox is not None:
                    x1, y1, x2, y2 = head_bbox
                    # Đảm bảo bbox nằm trong giới hạn ảnh
                    h, w = person_roi.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    head_roi = person_roi[y1:y2, x1:x2]
                    if head_roi.size > 0:
                        roi_to_detect = head_roi
                        logger.info(f"Sử dụng headbox từ keypoints: ({x1},{y1},{x2},{y2})")
                    else:
                        head_bbox = None  # Reset nếu roi không hợp lệ
            
            # Detect face trên roi_to_detect (headbox hoặc full person)
            infos, _ = self.face_detection.detect(roi_to_detect)
            
            if not infos:
                # logger.info("Không tìm thấy khuôn mặt nào.")
                return None, 0.0
            
            # SỬA LẠI: Tìm dictionary có giá trị 'confidence' cao nhất
            best_face = max(infos, key=lambda x: x.get('confidence', 0.0))
            
            # SỬA LẠI: Lấy confidence và bbox từ dictionary best_face
            confidence = best_face.get('confidence', 0.0)
            bbox_obj = best_face.get('bbox')
            
            if bbox_obj is None:
                # logger.error("Khuôn mặt tốt nhất không có key 'bbox'.")
                return None, 0.0

            # Phần tính toán tọa độ với margin top
            h_roi, w_roi = roi_to_detect.shape[:2]
            
            # Tính margin top theo pixel
            face_height = bbox_obj.height * h_roi
            margin_top_px = face_height * top_margin
            
            x1_f = max(0, int(bbox_obj.xmin * w_roi))
            y1_f = max(0, int(bbox_obj.ymin * h_roi - margin_top_px))  # Trừ margin top
            x2_f = min(w_roi, int(x1_f + bbox_obj.width * w_roi))
            y2_f = min(h_roi, int(y1_f + face_height + margin_top_px))  # Cộng margin vào height
            
            # QUAN TRỌNG: Nếu dùng headbox, cần chuyển tọa độ về person_roi gốc
            if head_bbox is not None:
                x1_f += head_bbox[0]
                y1_f += head_bbox[1]
                x2_f += head_bbox[0]
                y2_f += head_bbox[1]
                logger.info(f"Adjusted face bbox to person_roi: ({x1_f},{y1_f},{x2_f},{y2_f})")
            
            if x2_f <= x1_f or y2_f <= y1_f:
                # logger.warning("Kích thước bbox của khuôn mặt không hợp lệ.")
                return None, 0.0
                
            return np.array([x1_f, y1_f, x2_f, y2_f]), float(confidence)

        except Exception as e:
            logger.error(f"Lỗi trong _detect_best_face: {e}", exc_info=True)
            return None, 0.0
    
    def _is_pose_reliable(self, bbox, keypoints=None):
        """Kiểm tra độ tin cậy của pose"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            return False
        
        area = w * h
        if area < MIN_PERSON_AREA:
            return False
        
        aspect_ratio = h / w
        if aspect_ratio < MIN_ASPECT_RATIO_FOR_POSE or aspect_ratio > MAX_ASPECT_RATIO_FOR_POSE:
            return False
        
        # Kiểm tra số lượng keypoints đáng tin cậy
        if keypoints is not None and keypoints.shape[0] >= 17 and keypoints.shape[1] >= 3:
            # Đếm số keypoints có confidence > 0.3
            valid_keypoints_count = np.sum(keypoints[:, 2] > 0.7)
            if valid_keypoints_count < MIN_KEYPOINTS_FOR_POSE:
                logger.info(f"Không đủ keypoints cho pose model: {valid_keypoints_count} < {MIN_KEYPOINTS_FOR_POSE}")
                return False
        
        return True
    
    def _predict_face(self, face_roi):
        """Dự đoán từ khuôn mặt"""
        try:
            results = self.gender_face_model(face_roi, verbose=False)
            if not results or len(results) == 0:
                return np.zeros(len(self.face_classes))
            
            result = results[0]
            
            # Classification model
            if hasattr(result, 'probs') and result.probs is not None:
                return result.probs.data.cpu().numpy()
            
            # Detection model
            elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                max_idx = np.argmax(confidences)
                confidence = float(confidences[max_idx])
                pred_class = int(result.boxes.cls.cpu().numpy()[max_idx])
                
                probs = np.zeros(len(self.face_classes))
                if pred_class < len(self.face_classes):
                    probs[pred_class] = confidence
                return probs
            
            return np.zeros(len(self.face_classes))
        except:
            return np.zeros(len(self.face_classes))
    
    def _predict_pose(self, person_roi):
        """Dự đoán từ pose"""
        try:
            results = self.gender_pose_model(person_roi, verbose=False)
            if not results or len(results) == 0:
                return np.zeros(len(self.pose_classes))
            
            result = results[0]
            
            # Classification model
            if hasattr(result, 'probs') and result.probs is not None:
                return result.probs.data.cpu().numpy()
            
            # Detection model  
            elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                max_idx = np.argmax(confidences)
                confidence = float(confidences[max_idx])
                pred_class = int(result.boxes.cls.cpu().numpy()[max_idx])
                
                probs = np.zeros(len(self.pose_classes))
                if pred_class < len(self.pose_classes):
                    probs[pred_class] = confidence
                return probs
                   
            return np.zeros(len(self.pose_classes))
        except:
            return np.zeros(len(self.pose_classes))