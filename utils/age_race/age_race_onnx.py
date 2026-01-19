# utils/age_race_onnx.py
import cv2
import numpy as np
import os
import onnxruntime as ort
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class AgeRaceEstimatorONNX:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"❌ [ONNX] Không tìm thấy file model: {model_path}")
            self.session = None
            return

        logger.info(f"🔄 [ONNX] Đang tải model Age/Race: {model_path}...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_height = input_shape[2] if len(input_shape) == 4 else 224
            self.input_width = input_shape[3] if len(input_shape) == 4 else 224
            
            logger.info(f"✅ [ONNX] Khởi tạo thành công! Input: {self.input_width}x{self.input_height}")
        except Exception as e:
            logger.error(f"❌ [ONNX] Lỗi khởi tạo session: {e}")
            self.session = None

        self.age_labels = ["0-10", "11-19", "20-30", "31-40", "41-50", "50-69", "70+"]
        self.race_labels = ["White", "Black", "Asian", "Indian", "Others"]  # ✅ ĐỔI THỨ TỰ GIỐNG CODE TEST
        
        # ✅ THÊM IMAGENET NORMALIZATION
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, face_img):
        """
        ✅ CHUẨN HÓA GIỐNG CODE TEST 100%
        """
        if face_img is None or face_img.size == 0:
            return None
        
        try:
            # 1. Resize
            img = cv2.resize(face_img, (self.input_width, self.input_height))
            
            # 2. BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # 4. ✅ ImageNet Standardization (BƯỚC QUAN TRỌNG NHẤT!)
            img = (img - self.mean) / self.std
            
            # 5. HWC → CHW
            img = img.transpose(2, 0, 1)
            
            # 6. Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
        except Exception as e:
            logger.error(f"Lỗi preprocess: {e}")
            return None

    def predict(self, face_img):
        if self.session is None or face_img is None or face_img.size == 0:
            return None

        try:
            def softmax(x):
                x = np.asarray(x)
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            
            input_data = self.preprocess(face_img)
            if input_data is None:
                return None
            
            outputs = self.session.run(None, {self.input_name: input_data})
            
            # ✅ XỬ LÝ OUTPUT (Giống code test)
            age_idx = np.argmax(outputs[0])
            race_idx = np.argmax(outputs[1])
            
            # ✅ Tính softmax để có confidence
            age_probs = softmax(outputs[0][0])
            race_probs = softmax(outputs[1][0])
            
            age_conf = float(age_probs[age_idx])
            race_conf = float(race_probs[race_idx])
            
            age_label = self.age_labels[age_idx]
            race_label = self.race_labels[race_idx]
            
            # ✅ LOG DEBUG
            print(f"🧪 [AGE/RACE ONNX] Age: {age_label} ({age_conf:.3f}) | Race: {race_label} ({race_conf:.3f})")

            return {
                "age": age_label,
                "age_conf": age_conf,
                "race": race_label,
                "race_conf": race_conf
            }
        except Exception as e:
            logger.error(f"Lỗi predict ONNX: {e}")
            return None