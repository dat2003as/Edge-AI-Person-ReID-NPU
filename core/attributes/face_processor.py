# core/attributes/face_processor.py
import cv2
import numpy as np
import logging
import mediapipe as mp
from skimage import transform as trans

logger = logging.getLogger(__name__)

class FaceProcessor:
    """
    Class tiện ích (Utility) chuyên xử lý hình ảnh khuôn mặt và tính toán hình học.
    Không chứa state của model AI.
    """
    def __init__(self, detector=None):
        # Sử dụng instance CenterFace được truyền vào từ ModelsHandler
        self.detector = detector
        self.mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
    def detect_face(self, image, threshold=0.4):
        """Sử dụng CenterFace để tìm mặt và landmarks"""
        if self.detector is None:
            return None, None
        return self.detector.detect(image, threshold=threshold)
    
    MODEL_POINTS_5 = np.array([
        ( -225.0,  170.0, -135.0), # Mắt trái
        (  225.0,  170.0, -135.0), # Mắt phải
        (    0.0,    0.0,    0.0), # Mũi
        ( -150.0, -150.0, -125.0), # Miệng trái
        (  150.0, -150.0, -125.0)  # Miệng phải
    ], dtype=np.float32)

    @staticmethod
    def check_image_quality(image, min_size=(64, 64), blur_threshold=25.0,dark_threshold=25):
        """
        Kiểm tra chất lượng ảnh mặt: độ nét và độ sáng.
        Trả về False nếu ảnh quá mờ hoặc quá tối.
        """
        if image is None or image.size == 0: return False
        h, w = image.shape[:2]
        # 1. Check size
        if w < min_size[0] or h < min_size[1]: 
            return False
        
        # 2. Check blur (Laplacian Variance)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score < blur_threshold:
                return False # Ảnh quá mờ
            # 2. --- 🔹 KIỂM TRA ĐỘ SÁNG (BRIGHTNESS) MỚI 🔹 ---
            # Tính độ sáng trung bình của ảnh xám
            mean_brightness = np.mean(gray)
            # Nếu độ sáng thấp hơn ngưỡng (ví dụ 50 trên thang 255), coi là quá tối
            if mean_brightness < dark_threshold:
                # logger.warning(f"Ảnh quá tối (Brightness: {mean_brightness:.2f} < {dark_threshold}). Bỏ qua.")
                return False # Loại bỏ ngay lập tức

        except:
            return False       
        return True
    @staticmethod
    def safe_crop(image, x1, y1, x2, y2, tag="Unknown"):
        if image is None or image.size == 0: return None
        h, w = image.shape[:2]
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1: return None
        return image[y1:y2, x1:x2]

    @staticmethod
    def get_face_with_margin(person_crop, face_bbox, margin_ratio=0.1):
        try:
            h, w = person_crop.shape[:2]
            fx1, fy1, fx2, fy2 = map(int, face_bbox)
            fw, fh = fx2 - fx1, fy2 - fy1
            
            side = max(fw, fh)
            margin = int(side * margin_ratio)
            
            cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            
            half_side = (side // 2) + margin
            nx1, ny1 = max(0, cx - half_side), max(0, cy - half_side)
            nx2, ny2 = min(w, cx + half_side), min(h, cy + half_side)
            
            face_img = person_crop[ny1:ny2, nx1:nx2]
            
            return face_img, (nx1, ny1) 
        except Exception as e:
            logger.error(f"Lỗi trong get_face_with_margin: {e}")
            return None, (0, 0)

    @staticmethod
    def check_face_straight_2d(lms):
        # lms shape (5, 2): 0:Mắt trái, 1:Mắt phải, 2:Mũi
        try:
            # Kiểm tra Yaw (Xoay trái/phải)
            dist_l = np.linalg.norm(lms[2] - lms[0])
            dist_r = np.linalg.norm(lms[2] - lms[1])
            # Tỉ lệ càng lớn -> Mặt càng nghiêng
            yaw_ratio = max(dist_l, dist_r) / (min(dist_l, dist_r) + 1e-6)
            
            # Kiểm tra Roll (Nghiêng đầu vai)
            eye_diff_y = abs(lms[1][1] - lms[0][1])
            eye_dist = np.linalg.norm(lms[1] - lms[0])
            roll_ratio = eye_diff_y / (eye_dist + 1e-6)

            # Ngưỡng: yaw_ratio < 1.5 là khá thẳng. > 2.0 là nghiêng nhiều.
            is_straight = yaw_ratio < 1.8 and roll_ratio < 0.2
            return is_straight, yaw_ratio
        except:
            return False, 99.0
        
    @staticmethod
    def align_face_2d(image, lms, output_size=112):
        """
        Sửa đổi: Thêm tham số output_size để tùy biến 112 hoặc 224.
        """
        try:
            # Tọa độ chuẩn cho 112x112
            base_dst = np.array([
                [38.2946, 51.6963], [73.5318, 51.5014], # Mắt
                [56.0252, 71.7366],                     # Mũi
                [41.5493, 92.3655], [70.7299, 92.2041]  # Miệng
            ], dtype=np.float32)

            # Tỉ lệ hóa tọa độ chuẩn theo size mới
            ratio = output_size / 112.0
            dst_points = base_dst * ratio

            src_points = np.array(lms, dtype=np.float32)
            
            from skimage import transform as trans
            tform = trans.SimilarityTransform()
            tform.estimate(src_points, dst_points)
            M = tform.params[0:2, :]

            # Warp ảnh theo size mong muốn
            return cv2.warpAffine(image, M, (output_size, output_size), borderValue=0)
        except Exception as e:
            # Fallback nếu lỗi transform
            return cv2.resize(image, (output_size, output_size))

    @staticmethod
    def calculate_simple_golden_score(lms):
        try:
            # Tỉ lệ 1: Độ cân đối trái phải (Mắt-Mũi)
            dist_l = np.linalg.norm(lms[2] - lms[0])
            dist_r = np.linalg.norm(lms[2] - lms[1])
            balance_score = 100 - abs(dist_l - dist_r) / (dist_l + dist_r) * 100
            
            # Tỉ lệ 2: Khoảng cách mắt so với độ rộng mặt (giả định qua 5 điểm)
            eye_dist = np.linalg.norm(lms[1] - lms[0])
            mouth_dist = np.linalg.norm(lms[4] - lms[3])
            # Một tỉ lệ khuôn mặt đẹp thường có eye_dist / mouth_dist ~ 1.2
            ratio_val = eye_dist / (mouth_dist + 1e-6)
            ratio_score = 100 - abs(ratio_val - 1.2) * 50
            
            final_score = (balance_score + ratio_score) / 2
            return round(max(0, min(100, final_score)), 1)
        except:
            return 0
        
    @staticmethod
    def remove_background(face_img, bg_color=(128, 128, 128)):
        """
        Chuyển background vùng mặt về màu xám trung tính.
        """
        try:
            # Chuyển BGR sang RGB cho MediaPipe
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            results = FaceProcessor.mp_selfie.process(img_rgb)
            
            # Tạo mặt nạ (mask)
            condition = results.segmentation_mask > 0.5
            condition = np.stack((condition,) * 3, axis=-1)
            
            # Tạo nền phẳng
            bg_image = np.zeros(face_img.shape, dtype=np.uint8)
            bg_image[:] = bg_color
            
            # Kết hợp mặt và nền mới
            output_image = np.where(condition, face_img, bg_image)
            return output_image
        except:
            return face_img
    @staticmethod
    def align_face_224(image, lms):
        """
        Align mặt về kích thước 224x224 sử dụng 5 landmarks từ YuNet.
        
        Args:
            image: ảnh gốc (BGR, numpy array)
            lms: list hoặc array 10 phần tử theo thứ tự YuNet:
                [x_right_eye, y_right_eye, x_left_eye, y_left_eye,
                x_nose, y_nose, x_right_mouth, y_right_mouth,
                x_left_mouth, y_left_mouth]
        
        Returns:
            aligned_face: ảnh đã align 224x224 (hoặc None nếu lỗi)
        """
        try:
            # Tọa độ chuẩn cố định cho 224x224 (từ InsightFace/ArcFace)
            dst_points = np.array([
                [76.5892, 103.3926],   # right eye
                [147.0636, 103.0028],  # left eye
                [112.0504, 143.4732],  # nose
                [83.0986, 184.7310],   # right mouth corner
                [141.4598, 184.4082]   # left mouth corner
            ], dtype=np.float32)

            # Chuyển landmarks detect được thành array 5x2
            src_points = np.array(lms, dtype=np.float32).reshape(5, 2)

            # Ước lượng similarity transform (xoay + scale + dịch, giữ tỷ lệ mặt tốt hơn)
            tform = trans.SimilarityTransform()
            tform.estimate(src_points, dst_points)
            M = tform.params[0:2, :]  # Ma trận 2x3

            # Warp ảnh về đúng 224x224
            aligned = cv2.warpAffine(
                image, M, (224, 224),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,  # fill đen vùng ngoài
                flags=cv2.INTER_LINEAR
            )

            return aligned

        except Exception as e:
            print(f"Alignment failed: {e}")
            # Fallback: crop trung tâm và resize (nếu landmarks lỗi nặng)
            h, w = image.shape[:2]
            size = min(h, w)
            cx, cy = w // 2, h // 2
            crop = image[cy - size//2 : cy + size//2, cx - size//2 : cx + size//2]
            return cv2.resize(crop, (224, 224))