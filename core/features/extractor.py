# core/features/extractor.py
import torch
import torchreid
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os 
import logging

logger = logging.getLogger(__name__)

# --- Cấu hình đường dẫn ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
MODELS_DIR = os.path.join(project_root, 'models')

# OSNet (Re-ID)
OSNET_MODEL_PATH = os.path.join(MODELS_DIR, "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0_0015_coslr_b64_fb10.pth")
OSNET_INPUT_SIZE = (128, 256)

# MobileFaceNet (Face Recognition)
MOBILEFACENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilefacenet.pt") 
MOBILEFACENET_INPUT_SIZE = (112, 112)

class Analyzer:
    """
    Class chuyên trích xuất đặc trưng:
    - OSNet cho Re-ID (Toàn thân)
    - MobileFaceNet cho Face Recognition (Khuôn mặt)
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.profiler = None 
        
        print(f"✅ Analyzer khởi chạy trên: {self.device}")

        try:
            # 1. Tải model Re-ID
            self._load_osnet_model()

            # 2. Tải model MobileFaceNet
            self._load_mobilefacenet_model()

        except Exception as e:
            print(f"❌ Lỗi khởi tạo Analyzer: {e}")
            raise

   
    def _load_osnet_model(self):
        """Tải model OSNet."""
        print("Đang tải model OSNet...")
        self.osnet_model = torchreid.models.build_model(
            name='osnet_ain_x1_0', num_classes=1000, loss='softmax', pretrained=False
        )
        torchreid.utils.load_pretrained_weights(self.osnet_model, OSNET_MODEL_PATH)
        self.osnet_model.to(self.device)
        self.osnet_model.eval()
        _, self.osnet_transform = torchreid.reid.data.transforms.build_transforms(
            height=OSNET_INPUT_SIZE[1], width=OSNET_INPUT_SIZE[0], is_train=False
        )

    def _load_mobilefacenet_model(self):
        """Tải model MobileFaceNetV2."""
        print("Đang tải model MobileFaceNetV2...")
        if not os.path.exists(MOBILEFACENET_MODEL_PATH):
            raise FileNotFoundError(f"Thiếu file: {MOBILEFACENET_MODEL_PATH}")

        self.face_model = torch.jit.load(MOBILEFACENET_MODEL_PATH, map_location=self.device)
        self.face_model.to(self.device)
        self.face_model.eval()
        print("✅ Tải MobileFaceNetV2 thành công.")

    def extract_reid_feature(self, person_crop: np.ndarray, body_mask: np.ndarray = None) -> list | None:
        """Trích xuất vector đặc trưng Re-ID."""
        if person_crop is None or person_crop.size == 0:
            return None
        try:
            input_crop = person_crop
            if body_mask is not None:
                # Resize mask cho khớp ảnh crop
                mask_resized = cv2.resize(body_mask, (person_crop.shape[1], person_crop.shape[0]))
                # Chỉ giữ lại phần người (nền thành đen tuyệt đối)
                input_crop = cv2.bitwise_and(person_crop, person_crop, mask=mask_resized)

            rgb_crop = cv2.cvtColor(input_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_crop)
            transformed_image = self.osnet_transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.osnet_model(transformed_image)
            
            # Chuẩn hóa L2 cho Re-ID
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Lỗi Re-ID: {e}")
            return None

        
    def extract_face_feature(self, face_crop: np.ndarray) -> tuple[list | None, float]:
        """Trích xuất Face Vector bằng MobileFaceNet với CLAHE & L2 Norm."""
        if face_crop is None or face_crop.size == 0:
            return None, 0.0
        
        if self.profiler: self.profiler.start("Face_MobileFaceNet")
        
        try:
            # 1. Tiền xử lý CLAHE
            face_ready = face_crop
            # 2. Resize & Normalize chuẩn MobileFaceNet
            img_rgb = cv2.cvtColor(face_ready, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (112, 112))
            img_normalized = (img_resized.astype(np.float32) - 127.5) / 128.0
            
            # 3. Chuyển tensor & đưa lên device
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
            transformed_image = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.face_model(transformed_image)
            # 4. 🔥 CHUẨN HÓA L2 (Tăng điểm số similarity)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            if self.profiler: self.profiler.stop("Face_MobileFaceNet")
            return embedding.cpu().numpy().flatten().tolist(), 1.0 

        except Exception as e:
            if self.profiler: self.profiler.stop("Face_MobileFaceNet")
            print(f"❌ Lỗi Face Feature: {e}")
            return None, 0.0