# utils/gender_cix.py
import cv2
import numpy as np
import os
import sys
import time
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import config
except ImportError:
    config = None

from utils.NOE_Engine import EngineInfer
from utils.logging_python_orangepi import get_logger
from utils.detectors.mdp_face_detection_task import FaceDetection
from utils.detectors.cut_body_part import extract_head_from_frame

logger = get_logger(__name__)

FACE_CONFIDENCE_HIGH_THRESHOLD = 0.8
FACE_CONFIDENCE_LOW_THRESHOLD = 0.6

FACE_WEIGHT = 0.6
POSE_WEIGHT = 1 - FACE_WEIGHT

DEFAULT_POSE_CONFIDENCE = 0.8
MIN_PERSON_AREA = 150 * 100
MAX_ASPECT_RATIO_FOR_POSE = 5
MIN_ASPECT_RATIO_FOR_POSE = 1.5

MIN_POSE_CONFIDENCE_FOR_COMBINE = 0.65
MIN_KEYPOINTS_FOR_POSE = 27

CLASS_NAME = ['female', 'male']

class GenderClassification:
    def __init__(self, gender_face_model_path, gender_pose_model_path, device='cpu'):
        """
        Khoi tao he thong nhan dien gioi tinh voi NPU CIX models
        Args:
            gender_face_model_path: Duong dan model face CIX
            gender_pose_model_path: Duong dan model pose CIX
            device: 'cpu' hoac 'cuda' (khong dung cho NPU)
        """
        self.device = device
        
        self.face_model_path = gender_face_model_path
        self.pose_model_path = gender_pose_model_path
        
        self.face_detection = FaceDetection(min_detection_confidence=0.5)
        
        self.face_classes = CLASS_NAME
        self.pose_classes = CLASS_NAME        
        self.save_dir = None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"Anh nhan dien gioi tinh se duoc luu tai: {self.save_dir}")
        
        logger.info(f"Dang tai Gender Face CIX model: {self.face_model_path}...")
        if not os.path.exists(self.face_model_path):
            logger.error(f"Khong tim thay model Face CIX tai: {self.face_model_path}")
            self.gender_face_model = None
        else:
            try:
                self.gender_face_model = EngineInfer(self.face_model_path)
                logger.info(f"[CIX] Khoi tao Gender Face thanh cong! Model: {os.path.basename(self.face_model_path)}")
            except Exception as e:
                logger.error(f"[CIX] Loi khoi tao Face EngineInfer: {e}")
                self.gender_face_model = None
        
        logger.info(f"Dang tai Gender Pose CIX model: {self.pose_model_path}...")
        if not os.path.exists(self.pose_model_path):
            logger.error(f"Khong tim thay model Pose CIX tai: {self.pose_model_path}")
            self.gender_pose_model = None
        else:
            try:
                self.gender_pose_model = EngineInfer(self.pose_model_path)
                logger.info(f"[CIX] Khoi tao Gender Pose thanh cong! Model: {os.path.basename(self.pose_model_path)}")
            except Exception as e:
                logger.error(f"[CIX] Loi khoi tao Pose EngineInfer: {e}")
                self.gender_pose_model = None
        
        print("GenderClassification CIX initialized successfully")
        
        self.input_size = (224, 224)

    def preprocess(self, img):
        """
        Preprocessing cho Gender model (KHÔNG dùng ImageNet normalization)
        - Resize to 224x224
        - BGR to RGB
        - Normalize [0, 1]
        - HWC to CHW
        - Add batch dimension
        """
        if img is None or img.size == 0:
            return None
        
        try:
            img_resized = cv2.resize(img, self.input_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            img_chw = np.transpose(img_float, (2, 0, 1))
            img_batch = np.expand_dims(img_chw, axis=0)
            return img_batch
        except Exception as e:
            logger.error(f"Loi preprocess gender: {e}")
            return None
   
    def predict(self, human_img, keypoints=None):
        """
        Ham chinh de nhan dien gioi tinh tu anh nguoi da crop
        
        Returns:
            Dictionary voi format moi:
            {
                "gender": "male"/"female"/"unknown",
                "confidence": float (0-1),
                "strategy": "face_only"/"combined"/"pose_only"/"unknown"
            }
        """
        if human_img is None or human_img.size == 0:
            return {"gender": "unknown", "confidence": 0.0, "strategy": "unknown"}
        
        h, w = human_img.shape[:2]
        pseudo_bbox = np.array([0, 0, w, h])
        
        result = self._classify_internal(human_img, pseudo_bbox, keypoints)
        
        gender = result.get("gender", "Unknown").lower()
        if gender == "unknown":
            gender = "unknown"

        if self.save_dir:
            self._save_gender_image(human_img, gender)

        logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info(f"Gender: {gender}, Confidence: {result.get('confidence', 0.0):.2f}, Strategy: {result.get('strategy', 'unknown')}")
        
        return {
            "gender": gender,
            "confidence": result.get("confidence", 0.0),
            "strategy": result.get("strategy", "unknown")
        }

    def _save_gender_image(self, image: np.ndarray, gender: str):
        """
        Luu anh cua nguoi vao thu muc con tuong ung voi gioi tinh.
        
        Args:
            image (np.ndarray): Anh bbox cua nguoi.
            gender (str): Gioi tinh da duoc du doan ('male', 'female', 'unknown').
        """
        if not self.save_dir:
            return

        try:
            gender_folder = gender if gender in ['male', 'female'] else 'unknown'
            
            target_dir = os.path.join(self.save_dir, gender_folder)
            os.makedirs(target_dir, exist_ok=True)
            
            filename = f"{int(time.time() * 1000)}.jpg"
            filepath = os.path.join(target_dir, filename)
            
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            logger.info(f"Da luu anh gioi tinh vao: {filepath}")

        except Exception as e:
            logger.error(f"Khong the luu anh gioi tinh: {e}", exc_info=True)

    def _classify_internal(self, person_roi, person_bbox, keypoints=None):
        """
        Logic nhan dien noi bo
        """
        face_bbox, face_confidence = self._detect_best_face(person_roi, keypoints)
        logger.info(f"face_bbox: {face_bbox}, face_confidence: {face_confidence}")
        
        if face_confidence >= FACE_CONFIDENCE_HIGH_THRESHOLD:
            strategy = "face_only"
        elif face_confidence >= FACE_CONFIDENCE_LOW_THRESHOLD:
            strategy = "combined"
        else:
            strategy = "pose_only"
        logger.info(f"strategy: {strategy}")
        result = {"gender": "Unknown", "confidence": 0.0, "strategy": strategy}
        
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
                if face_bbox is not None:
                    x1_f, y1_f, x2_f, y2_f = map(int, face_bbox)
                    face_roi = person_roi[y1_f:y2_f, x1_f:x2_f]
                    if face_roi.size > 0:
                        face_probs = self._predict_face(face_roi)
                        pred_idx = np.argmax(face_probs)
                        result["gender"] = self.face_classes[pred_idx]
                        result["confidence"] = float(face_probs[pred_idx])
            else:
                face_probs = np.zeros(len(self.face_classes))
                if face_bbox is not None:
                    x1_f, y1_f, x2_f, y2_f = map(int, face_bbox)
                    face_roi = person_roi[y1_f:y2_f, x1_f:x2_f]
                    if face_roi.size > 0:
                        face_probs = self._predict_face(face_roi)
                
                pose_probs = self._predict_pose(person_roi)
                
                pose_max_confidence = float(np.max(pose_probs))
                if pose_max_confidence < MIN_POSE_CONFIDENCE_FOR_COMBINE:
                    if np.max(face_probs) > 0:
                        pred_idx = np.argmax(face_probs)
                        result["gender"] = self.face_classes[pred_idx]
                        result["confidence"] = float(face_probs[pred_idx])
                else:
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
        """Phat hien khuon mat tot nhat voi preprocessing headbox neu co keypoints"""
        try:
            roi_to_detect = person_roi
            head_bbox = None
            
            if keypoints is not None and keypoints.shape[0] >= 7:
                head_bbox = extract_head_from_frame(person_roi, keypoints, scale=1.2)
                
                if head_bbox is not None:
                    x1, y1, x2, y2 = head_bbox
                    h, w = person_roi.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    head_roi = person_roi[y1:y2, x1:x2]
                    if head_roi.size > 0:
                        roi_to_detect = head_roi
                        logger.info(f"Su dung headbox tu keypoints: ({x1},{y1},{x2},{y2})")
                    else:
                        head_bbox = None
            
            infos, _ = self.face_detection.detect(roi_to_detect)
            
            if not infos:
                return None, 0.0
            
            best_face = max(infos, key=lambda x: x.get('confidence', 0.0))
            
            confidence = best_face.get('confidence', 0.0)
            bbox_obj = best_face.get('bbox')
            
            if bbox_obj is None:
                return None, 0.0

            h_roi, w_roi = roi_to_detect.shape[:2]
            
            face_height = bbox_obj.height * h_roi
            margin_top_px = face_height * top_margin
            
            x1_f = max(0, int(bbox_obj.xmin * w_roi))
            y1_f = max(0, int(bbox_obj.ymin * h_roi - margin_top_px))
            x2_f = min(w_roi, int(x1_f + bbox_obj.width * w_roi))
            y2_f = min(h_roi, int(y1_f + face_height + margin_top_px))
            
            if head_bbox is not None:
                x1_f += head_bbox[0]
                y1_f += head_bbox[1]
                x2_f += head_bbox[0]
                y2_f += head_bbox[1]
                logger.info(f"Adjusted face bbox to person_roi: ({x1_f},{y1_f},{x2_f},{y2_f})")
            
            if x2_f <= x1_f or y2_f <= y1_f:
                return None, 0.0
                
            return np.array([x1_f, y1_f, x2_f, y2_f]), float(confidence)

        except Exception as e:
            logger.error(f"Loi trong _detect_best_face: {e}", exc_info=True)
            return None, 0.0
    
    def _is_pose_reliable(self, bbox, keypoints=None):
        """Kiem tra do tin cay cua pose"""
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
        
        if keypoints is not None and keypoints.shape[0] >= 17 and keypoints.shape[1] >= 3:
            valid_keypoints_count = np.sum(keypoints[:, 2] > 0.7)
            if valid_keypoints_count < MIN_KEYPOINTS_FOR_POSE:
                logger.info(f"Khong du keypoints cho pose model: {valid_keypoints_count} < {MIN_KEYPOINTS_FOR_POSE}")
                return False
        
        return True
    
    def _predict_face(self, face_roi):
        """Du doan tu khuon mat voi NPU CIX"""
        try:
            if self.gender_face_model is None:
                logger.error("Gender Face model chua duoc load")
                return np.zeros(len(self.face_classes))
            
            input_data = self.preprocess(face_roi)
            if input_data is None:
                return np.zeros(len(self.face_classes))
            
            outputs = self.gender_face_model.forward([input_data])
            
            if outputs is None or len(outputs) == 0:
                logger.error(f"Face output khong hop le: {type(outputs)}")
                return np.zeros(len(self.face_classes))
            
            logits = outputs[0]
            
            if isinstance(logits, np.ndarray) and logits.ndim > 1:
                logits = logits.flatten()
            
            def softmax(x):
                x = np.asarray(x)
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            
            probs = softmax(logits)
            
            logger.debug(f"[FACE CIX] Probs: {probs}")
            
            return probs
        except Exception as e:
            logger.error(f"Loi predict face CIX: {e}", exc_info=True)
            return np.zeros(len(self.face_classes))
    
    def _predict_pose(self, person_roi):
        """Du doan tu pose voi NPU CIX"""
        try:
            if self.gender_pose_model is None:
                logger.error("Gender Pose model chua duoc load")
                return np.zeros(len(self.pose_classes))
            
            input_data = self.preprocess(person_roi)
            if input_data is None:
                return np.zeros(len(self.pose_classes))
            
            outputs = self.gender_pose_model.forward([input_data])
            
            if outputs is None or len(outputs) == 0:
                logger.error(f"Pose output khong hop le: {type(outputs)}")
                return np.zeros(len(self.pose_classes))
            
            logits = outputs[0]
            
            if isinstance(logits, np.ndarray) and logits.ndim > 1:
                logits = logits.flatten()
            
            def softmax(x):
                x = np.asarray(x)
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            
            probs = softmax(logits)
            
            logger.debug(f"[POSE CIX] Probs: {probs}")
            
            return probs
        except Exception as e:
            logger.error(f"Loi predict pose CIX: {e}", exc_info=True)
            return np.zeros(len(self.pose_classes))
    
    def release(self):
        """Giai phong NPU resource"""
        if self.gender_face_model:
            try:
                self.gender_face_model.clean()
                logger.info("Face model cleaned successfully")
            except Exception as e:
                logger.error(f"Loi clean face model: {e}")
        
        if self.gender_pose_model:
            try:
                self.gender_pose_model.clean()
                logger.info("Pose model cleaned successfully")
            except Exception as e:
                logger.error(f"Loi clean pose model: {e}")
