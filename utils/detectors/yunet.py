#utils/centerface.py
import cv2
import numpy as np

class YuNetDetector:
    def __init__(self, model_path, input_size=(640, 640), score_threshold=0.6, nms_threshold=0.3):
        """
        Khởi tạo YuNet Detector
        input_size: (width, height) - Càng nhỏ chạy trên Orange Pi càng nhanh.
        """
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=5000
        )
        self.input_size = input_size
        print(f"✅ YuNet 5-point Detector Loaded: {model_path}")

    def detect(self, image, threshold=None, profiler=None):
        if image is None or image.size == 0:
            return None, None

        if profiler: 
            profiler.start("Face_Detector (YuNet)")

        # --- BƯỚC 1: CẬP NHẬT KÍCH THƯỚC INPUT ---
        # YuNet yêu cầu input_size phải khớp với ảnh đưa vào
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))

        # --- BƯỚC 2: INFERENCE ---
        # YuNet tự động xử lý tiền xử lý và hậu xử lý bên trong
        result = self.detector.detect(image)
        faces = result[1]

        if faces is None:
            if profiler: profiler.stop("Face_Detector (YuNet)")
            return None, None

        # --- BƯỚC 3: TÁCH DETS VÀ LANDMARKS ---
        # faces là mảng [N, 15]
        # 0-3: bbox (x, y, w, h)
        # 4-13: 5 landmarks (x, y)
        # 14: score
        
        # Tạo dets theo format: [x1, y1, x2, y2, score] để giống CenterFace
        bboxes = faces[:, 0:4]
        scores = faces[:, 14:15]
        # Chuyển (x, y, w, h) -> (x1, y1, x2, y2)
        dets = np.hstack([bboxes[:, :2], bboxes[:, :2] + bboxes[:, 2:], scores])

        # Tách 5 landmarks: format [[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5], ...]
        landmarks = faces[:, 4:14]

        if profiler: 
            profiler.stop("Face_Detector (YuNet)")
            
        return dets, landmarks

    def release(self):
        # YuNet trong OpenCV tự giải phóng khi object bị hủy
        pass