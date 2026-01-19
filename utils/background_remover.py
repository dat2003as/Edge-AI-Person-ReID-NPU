import mediapipe as mp
import cv2 
import numpy as np
class BackgroundRemover:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentor = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0) # 0: general, 1: landscape

    def remove_bg(self, image):
        # Chuyển sang RGB cho MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmentor.process(img_rgb)
        
        # Tạo mask (mặt nạ)
        # kết quả trả về mask với giá trị từ 0.0 đến 1.0
        mask = results.segmentation_mask > 0.5 # Ngưỡng cắt
        
        # Chuyển mask sang định dạng OpenCV
        mask = mask.astype(np.uint8) * 255
        
        # Tạo ảnh kết quả với nền đen
        res = cv2.bitwise_and(image, image, mask=mask)
        return res
    @staticmethod
    def create_checkerboard_bg(shape, tile_size=20, color1=(220, 220, 220), color2=(150, 150, 150)):
        height, width = shape[:2]
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                color = color1 if (x // tile_size + y // tile_size) % 2 == 0 else color2
                cv2.rectangle(bg, (x, y), (x + tile_size, y + tile_size), color, -1)
        return bg