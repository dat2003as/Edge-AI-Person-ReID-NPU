# utils/age_race/age_race_cix.py
import cv2
import numpy as np
import os
import sys
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import config
except ImportError:
    config = None

from utils.NOE_Engine import EngineInfer
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class AgeRaceEstimator:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = getattr(config, 'AGE_RACE_MODEL_CIX_PATH', 'models/age_race_model_new_123.cix')
        
        self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            logger.error(f"Khong tim thay model CIX tai: {self.model_path}")
            self.model = None
            return

        logger.info(f"Dang tai model Age/Race CIX: {self.model_path}...")
        try:
            self.model = EngineInfer(self.model_path)
            logger.info(f"[CIX] Khoi tao thanh cong! Model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logger.error(f"[CIX] Loi khoi tao EngineInfer: {e}")
            self.model = None
            return

        self.age_labels = ["0-10", "11-19", "20-30", "31-40", "41-50", "51-69", "70+"]
        self.race_labels = ["White", "Black", "Asian", "Indian", "Others"]
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = (300, 300)

    def preprocess(self, face_img):
        """
        Preprocessing theo ImageNet standard (giong ONNX)
        - BGR to RGB
        - Normalize [0, 1]
        - ImageNet standardization
        - HWC to CHW
        - Add batch dimension
        """
        if face_img is None or face_img.size == 0:
            return None
        
        try:
            img = cv2.resize(face_img, self.input_size)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = img.astype(np.float32) / 255.0
            
            img = (img - self.mean) / self.std
            
            img = np.transpose(img, (2, 0, 1))
            
            img = np.expand_dims(img, axis=0)
            
            return img
        except Exception as e:
            logger.error(f"Loi preprocess: {e}")
            return None

    def predict(self, face_img):
        """
        Predict age and race from face image
        
        Args:
            face_img: OpenCV BGR image
            
        Returns:
            dict: {
                'age': str (e.g., "20-30"),
                'age_conf': float,
                'race': str (e.g., "Asian"),
                'race_conf': float
            }
        """
        if self.model is None or face_img is None or face_img.size == 0:
            return None

        try:
            def softmax(x):
                x = np.asarray(x)
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            
            input_data = self.preprocess(face_img)
            if input_data is None:
                return None
            
            outputs = self.model.forward([input_data])
            
            if outputs is None or len(outputs) < 2:
                logger.error(f"Output khong hop le: {type(outputs)}")
                return None
            
            age_logits = outputs[0]
            race_logits = outputs[1]
            
            if isinstance(age_logits, np.ndarray) and age_logits.ndim > 1:
                age_logits = age_logits.flatten()
            if isinstance(race_logits, np.ndarray) and race_logits.ndim > 1:
                race_logits = race_logits.flatten()
            
            age_idx = np.argmax(age_logits)
            race_idx = np.argmax(race_logits)
            
            age_probs = softmax(age_logits)
            race_probs = softmax(race_logits)
            
            age_conf = float(age_probs[age_idx])
            race_conf = float(race_probs[race_idx])
            
            age_label = self.age_labels[age_idx]
            race_label = self.race_labels[race_idx]
            
            logger.debug(f"[AGE/RACE CIX] Age: {age_label} ({age_conf:.3f}) | Race: {race_label} ({race_conf:.3f})")

            return {
                "age": age_label,
                "age_conf": age_conf,
                "race": race_label,
                "race_conf": race_conf
            }
        except Exception as e:
            logger.error(f"Loi predict CIX: {e}")
            return None

    def release(self):
        """Giai phong NPU resource"""
        if self.model:
            try:
                self.model.clean()
                logger.info("[Age/Race CIX] Da giai phong model.")
            except Exception as e:
                logger.error(f"Loi khi clean model: {e}")
