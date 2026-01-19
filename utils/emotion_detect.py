import cv2
import numpy as np
import os
import sys
import torch

# Import cấu hình
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from config import MODEL_DIR
except ImportError:
    MODEL_DIR = os.path.join(BASE_DIR, 'models') 

# Import Ultralytics (Bắt buộc vì bạn đang dùng model .pt của YOLO)
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Lỗi: Chưa cài thư viện ultralytics. Hãy chạy: pip install ultralytics")
    YOLO = None

class EmotionEstimator:
    # Đường dẫn mặc định
    EMO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
    
    # Danh sách nhãn (Nếu model train đúng thứ tự này)
    # YOLOv8 thường lưu class names bên trong file model, ta sẽ ưu tiên lấy từ model trước.
    CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self, model_path=None, input_size=224, use_grayscale=False):
        """
        Khởi tạo Emotion Estimator sử dụng Ultralytics YOLO Engine.
        """
        self.model_path = model_path if model_path else self.EMO_MODEL_PATH
        self.model = None
        
        if YOLO is None:
            return

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"❌ [Emotion] Không tìm thấy file model tại: {self.model_path}")
            return

        print(f"--> Đang tải model YOLO Emotion từ: {self.model_path}")
        try:
            # --- CÁCH KHẮC PHỤC CHÍNH ---
            # Sử dụng class YOLO của ultralytics để load. 
            # Nó tự động xử lý việc file chỉ có weights (state_dict).
            self.model = YOLO(self.model_path)
            
            # Kiểm tra xem có đúng là model phân loại (Classify) không

            if hasattr(self.model, 'names'):
                print(f"✅ [Emotion] Đã tải thành công. Classes: {self.model.names}")
            else:
                print("✅ [Emotion] Đã tải model (không tìm thấy tên class, sẽ dùng mặc định).")
                
        except Exception as e:
            print(f"❌ [Emotion] Lỗi nghiêm trọng khi tải model YOLO: {e}")
            self.model = None

    def predict(self, face_image):
        """
        Dự đoán cảm xúc.
        Input: face_image (OpenCV BGR numpy array)
        """
        if self.model is None or face_image is None or face_image.size == 0:
            return None

        try:
            # YOLOv8 tự động xử lý Preprocessing (Resize, Normalize, BGR/RGB)
            # Nên ta chỉ cần truyền ảnh gốc vào.
            
            # verbose=False để tắt log spam console
            results = self.model(face_image, verbose=False)
            
            if not results:
                return None

            # Lấy kết quả đầu tiên (vì batch size = 1)
            r = results[0]
            
            # Lấy top 1 class
            # r.probs.top1 trả về index của class có xác suất cao nhất
            # r.probs.top1conf trả về độ tin cậy tương ứng
            idx = r.probs.top1
            conf = float(r.probs.top1conf)
            
            # Lấy tên nhãn
            # Ưu tiên lấy từ model (chính xác nhất theo lúc train)
            if hasattr(self.model, 'names') and idx in self.model.names:
                label = self.model.names[idx]
            else:
                # Fallback về danh sách cứng nếu model không có names
                if idx < len(self.CLASS_NAMES):
                    label = self.CLASS_NAMES[idx]
                else:
                    label = "Unknown"

            return {
                "emotion": label,
                "confidence": conf,
                "index": idx
            }

        except Exception as e:
            # print(f"⚠️ [Emotion] Lỗi khi predict: {e}")
            return None
