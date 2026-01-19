# utils/mediapipe_pose.py
import cv2
import numpy as np
import os
import time
import mediapipe as mp
from ultralytics import YOLO
from typing import Optional, List, Tuple, Dict

# Giả sử bạn có một module logging
from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)

mp_pose = mp.solutions.pose
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import RunningMode
        

# Các kết nối giữa các điểm keypoint của MediaPipe để vẽ skeleton
MEDIAPIPE_EDGES = [
    # Khuôn mặt
    (0, 1),   # Mũi -> Mắt trong trái
    (1, 2),   # Mắt trong trái -> Mắt trái
    (2, 3),   # Mắt trái -> Mắt ngoài trái
    (3, 7),   # Mắt ngoài trái -> Tai trái
    (0, 4),   # Mũi -> Mắt trong phải
    (4, 5),   # Mắt trong phải -> Mắt phải
    (5, 6),   # Mắt phải -> Mắt ngoài phải
    (6, 8),   # Mắt ngoài phải -> Tai phải
    (9, 10),  # Miệng trái -> Miệng phải

    # Thân trên
    (11, 12), # Vai trái -> Vai phải
    (11, 23), # Vai trái -> Hông trái
    (12, 24), # Vai phải -> Hông phải
    (23, 24), # Hông trái -> Hông phải

    # Tay trái
    (11, 13), # Vai trái -> Khuỷu tay trái
    (13, 15), # Khuỷu tay trái -> Cổ tay trái
    (15, 17), # Cổ tay trái -> Ngón út trái (pinky)
    (15, 19), # Cổ tay trái -> Ngón trỏ trái (index)
    (15, 21), # Cổ tay trái -> Ngón cái trái
    (17, 19), # Ngón út trái -> Ngón trỏ trái

    # Tay phải
    (12, 14), # Vai phải -> Khuỷu tay phải
    (14, 16), # Khuỷu tay phải -> Cổ tay phải
    (16, 18), # Cổ tay phải -> Ngón út phải (pinky)
    (16, 20), # Cổ tay phải -> Ngón trỏ phải (index)
    (16, 22), # Cổ tay phải -> Ngón cái phải
    (18, 20), # Ngón út phải -> Ngón trỏ phải

    # Chân trái
    (23, 25), # Hông trái -> Đầu gối trái
    (25, 27), # Đầu gối trái -> Mắt cá trái
    (27, 29), # Mắt cá trái -> Gót chân trái
    (27, 31), # Mắt cá trái -> Ngón chân cái trái
    (29, 31), # Gót chân trái -> Ngón chân cái trái

    # Chân phải
    (24, 26), # Hông phải -> Đầu gối phải
    (26, 28), # Đầu gối phải -> Mắt cá phải
    (28, 30), # Mắt cá phải -> Gót chân phải
    (28, 32), # Mắt cá phải -> Ngón chân cái phải
    (30, 32)  # Gót chân phải -> Ngón chân cái phải
]


class HumanDetection:
    """
    Class để phát hiện người và ước tính tư thế bằng cách kết hợp YOLO và MediaPipe.
    Các hàm tiện ích đã được chuyển thành staticmethod để dễ dàng tái sử dụng.
    """
    def __init__(self, person_model='models/yolo11n.pt', pose_model='models/pose_landmarker.task'):
        logger.info('Init Human Detection with YOLO + MediaPipe Pose (Buffer Mode)')
        self.classes = [0]
        
        # 1. Khởi tạo YOLO (Thư viện ultralytics xử lý file rất tốt)
        self.person_detector = YOLO(person_model)
        
        # --- 🔹 GIẢI PHÁP NẠP TỪ BUFFER (BỎ QUA LỖI ĐƯỜNG DẪN WINDOWS) 🔹 ---
        try:
            # Tự mở file bằng Python (Python xử lý mọi loại đường dẫn rất ổn định)
            if not os.path.exists(pose_model):
                raise FileNotFoundError(f"Không tìm thấy file model tại: {pose_model}")
            
            with open(pose_model, 'rb') as f:
                model_buffer = f.read()
            
            # Cấu hình MediaPipe
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                # Sử dụng model_asset_buffer thay vì model_asset_path
                base_options=BaseOptions(model_asset_buffer=model_buffer), 
                running_mode=VisionRunningMode.IMAGE,
                num_poses=1,
                output_segmentation_masks=True
            )
            
            # Khởi tạo landmarker từ dữ liệu trong RAM
            self.landmarker = PoseLandmarker.create_from_options(options)
            logger.info("✅ Pose Landmarker khởi tạo thành công bằng phương pháp Buffer.")

        except Exception as e:
            logger.error(f"❌ Lỗi nạp model MediaPipe: {e}")
            self.landmarker = None
        
        self.fps_avg = 0.0
        self.call_count = 0
        self.last_results = None

    # def __init__(self, person_model: str = 'python/models/yolo11.rknn', pose_model: str = 'python/face_processing/models/pose_landmarker.task'):
    #     """
    #     Khởi tạo lớp HumanDetection.

    #     Hàm này sẽ tải model YOLO để phát hiện người và model MediaPipe Pose Landmarker
    #     để phát hiện các điểm mốc tư thế.

    #     Args:
    #         person_model (str): Đường dẫn đến file model YOLO (.rknn).
    #         pose_model (str): Đường dẫn đến file model MediaPipe Pose Landmarker (.task).
    #     """
    #     logger.info('Initializing Human Detection with a hybrid YOLO and MediaPipe Pose approach...')

    #     # --- 1. Khởi tạo model YOLO để phát hiện người ---
    #     self.classes = [0]  # Lớp 0 thường là 'person' trong COCO dataset
    #     self.person_detector = YOLO(person_model)
    #     logger.info(f"Person detector (YOLO) initialized successfully with model: {person_model}")

    #     # --- 2. Khởi tạo MediaPipe Pose Landmarker ---
    #     # Ghi chú: Các lớp này nên được import ở đầu file để code sạch hơn

    #     # Tạo các tùy chọn cần thiết cho landmarker
    #     base_options = python.BaseOptions(model_asset_path=pose_model)
    #     options = vision.PoseLandmarkerOptions(
    #         base_options=base_options,
    #         running_mode=vision.RunningMode.IMAGE,
    #         num_poses=1,  # Tối ưu cho việc xử lý 1 người trong mỗi ảnh crop
    #         output_segmentation_masks=False
    #     )
        
    #     # Tạo landmarker từ các tùy chọn đã định nghĩa
    #     self.landmarker = vision.PoseLandmarker.create_from_options(options)
    #     logger.info(f"Pose Landmarker (MediaPipe) initialized successfully with model: {pose_model}")

    #     # --- 3. Khởi tạo các biến theo dõi hiệu suất ---
    #     self.fps_avg = 0.0
    #     self.call_count = 0
    #     self.last_results = None
    def detect_pose_from_bbox(self, full_frame: np.ndarray, bbox: tuple):
            """
            Ước tính tư thế cho một người duy nhất từ bounding box cho trước.
            """
            try:
                # 1. Trích xuất vùng ảnh của người đó từ frame gốc
                x1, y1, x2, y2 = map(int, bbox)
                padding = 10 # Thêm một chút đệm để đảm bảo không mất chi tiết
                person_crop_bgr = full_frame[max(0, y1-padding):y2+padding, max(0, x1-padding):x2+padding]

                if person_crop_bgr.size == 0:
                    return None, None

                # 2. Chạy MediaPipe Pose Landmarker trên vùng ảnh đã cắt
                person_crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=person_crop_rgb)
                detection_result = self.landmarker.detect(mp_image)


                person_keypoints = None
                person_z = None
                body_mask = None
                # 3. Chuyển đổi tọa độ keypoints (Quan trọng!)
                # MediaPipe trả về tọa độ tương đối (0-1) so với ảnh crop.
                # Chúng ta cần chuyển nó về tọa độ tuyệt đối so với frame gốc.
                if detection_result.pose_landmarks:
                    pose_landmarks = detection_result.pose_landmarks[0]
                    crop_h, crop_w, _ = person_crop_rgb.shape
                    
                    person_keypoints = np.zeros((33, 3))
                    person_z = np.zeros(33)

                    for i, lm in enumerate(pose_landmarks):
                        # Chuyển từ tọa độ tương đối (trong crop) sang tuyệt đối (trong crop)
                        local_x = lm.x * crop_w
                        local_y = lm.y * crop_h

                        # Chuyển từ tọa độ tuyệt đối (trong crop) sang tuyệt đối (trong frame gốc)
                        global_x = local_x + (x1 - padding)
                        global_y = local_y + (y1 - padding)

                        person_keypoints[i] = [global_x, global_y, lm.visibility]
                        person_z[i] = lm.z
                                    
                    if detection_result.segmentation_masks:
                        mask_data = detection_result.segmentation_masks[0].numpy_view()
                        # Chuyển đổi sang mask 0-255 để OpenCV dùng được
                        body_mask = (mask_data > 0.5).astype(np.uint8) * 255
                    
                    return person_keypoints, person_z, body_mask
                
                return None, None,None # Không tìm thấy pose
            except Exception as e:
                logger.error(f"Lỗi khi xử lý pose từ bbox: {e}")
                return None, None
    def run_detection(self, source: np.ndarray):
        start_time = time.time()
        image_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        
        # Phát hiện người bằng YOLO
        yolo_results = self.person_detector.predict(source=image_rgb, verbose=False, classes=self.classes, conf=0.5)
        logger.info(f"YOLO results: {len(yolo_results)} person detected")
        all_keypoints, all_z_values, boxes_data = [], [], []
        
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            boxes_data.append((x1, y1, x2, y2, conf))
            
            # Crop ảnh người với một chút padding
            padding = 10
            person_crop_rgb = image_rgb[max(0, y1-padding):min(image_rgb.shape[0], y2+padding), max(0, x1-padding):min(image_rgb.shape[1], x2+padding)]
            
            if person_crop_rgb.shape[0] == 0 or person_crop_rgb.shape[1] == 0:
                continue

            # Ước tính tư thế trên ảnh đã crop
            #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=person_crop_rgb)
            person_crop_rgb = np.ascontiguousarray(person_crop_rgb)
            mp_image = mp.Image(mp.ImageFormat.SRGB,person_crop_rgb)
            detection_result = self.landmarker.detect(mp_image)
            
            # Chuyển đổi tọa độ keypoint về hệ tọa độ của ảnh gốc
            person_keypoints, person_z = np.zeros((33, 3)), np.zeros(33)
            if detection_result.pose_landmarks:
                pose_landmarks = detection_result.pose_landmarks[0]
                crop_h, crop_w, _ = person_crop_rgb.shape
                for i, lm in enumerate(pose_landmarks):
                    abs_x = (lm.x * crop_w) + (x1 - padding)
                    abs_y = (lm.y * crop_h) + (y1 - padding)
                    person_keypoints[i] = [abs_x, abs_y, lm.visibility]
                    person_z[i] = lm.z
                    
            all_keypoints.append(person_keypoints)
            all_z_values.append(person_z)

        # Tính toán và ghi log FPS
        duration = time.time() - start_time
        fps_current = 1 / duration if duration > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1
        logger.info(f"FPS Human detection (YOLO+MediaPipe): {self.fps_avg:.2f}")

        # Trả về kết quả
        if not all_keypoints:
            self.last_results = None
            return np.array([]), [], np.array([])
            
        keypoints_data = np.array(all_keypoints)
        z_data = np.array(all_z_values)
        self.last_results = (keypoints_data, boxes_data)
        #logger.info(f"keypoints_data: {keypoints_data}")
        #logger.info(f"boxes_data: {boxes_data}")
        #logger.info(f"z_data: {z_data}")
        
        return keypoints_data, boxes_data, z_data

    # =====================================================================
    # ✨ CÁC HÀM TIỆN ÍCH ĐÃ ĐƯỢC CẬP NHẬT THÀNH STATICMETHOD
    # Giờ đây chúng ta có thể gọi chúng từ bên ngoài: HumanDetection.select_best_arm(...)
    # =====================================================================
    # Thêm hàm này vào class HumanDetection trong file mediapipe_pose.py

    @staticmethod
    def is_forearm_inside_torso(torso_poly: List[Tuple[int, int]], elbow: Dict, wrist: Dict) -> bool:
        """
        Kiểm tra xem đường thẳng từ khuỷu tay đến cổ tay có nằm bên trong đa giác thân áo không.
        Sử dụng cv2.pointPolygonTest để hiệu quả.
        """
        if not torso_poly or len(torso_poly) < 4 or elbow is None or wrist is None:
            return False

        try:
            # Chuyển đổi đa giác thành định dạng numpy cần thiết cho OpenCV
            contour = np.array(torso_poly, dtype=np.int32)
            
            # Kiểm tra cả hai điểm. Chỉ cần một điểm nằm trong hoặc trên biên là đủ.
            # Giá trị > 0: trong, == 0: trên biên, < 0: ngoài
            is_elbow_in = cv2.pointPolygonTest(contour, (int(elbow['x']), int(elbow['y'])), False) >= 0
            is_wrist_in = cv2.pointPolygonTest(contour, (int(wrist['x']), int(wrist['y'])), False) >= 0
            
            # Nếu một trong hai điểm nằm trong, ta coi là cẳng tay "bên trong"
            return is_elbow_in or is_wrist_in
        except Exception:
            return False

    @staticmethod
    def _get_limb_coords(side, limb_type, keypoints, z_coords):
        """[STATICMETHOD] Hàm nội bộ để lấy tọa độ của một chi cụ thể."""
        if limb_type == 'arm':
            indices = {
                'main': mp_pose.PoseLandmark[f'{side}_WRIST'].value,
                'p1': mp_pose.PoseLandmark[f'{side}_SHOULDER'].value,
                'p2': mp_pose.PoseLandmark[f'{side}_ELBOW'].value,
                'p3': mp_pose.PoseLandmark[f'{side}_WRIST'].value,
            }
            labels = ['shoulder', 'elbow', 'wrist']
        elif limb_type == 'leg':
            indices = {
                'main': mp_pose.PoseLandmark[f'{side}_ANKLE'].value,
                'p1': mp_pose.PoseLandmark[f'{side}_HIP'].value,
                'p2': mp_pose.PoseLandmark[f'{side}_KNEE'].value,
                'p3': mp_pose.PoseLandmark[f'{side}_ANKLE'].value,
            }
            labels = ['hip', 'knee', 'ankle']
        else:
            return None

        coords = {}
        for i, label in enumerate(labels):
            idx = indices[f'p{i+1}']
            x, y, vis = keypoints[idx]
            z = z_coords[idx]
            coords[label] = {'x': x, 'y': y, 'z': z, 'visibility': vis}
        return coords

    @staticmethod
    def select_best_arm(keypoints: np.ndarray, z_coords: np.ndarray, visibility_threshold: float = 0.9):
        """
        [STATICMETHOD] Chọn cánh tay tốt nhất dựa trên visibility và khoảng cách Z (giá trị Z nhỏ hơn là gần camera hơn).
        """
        left_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
        right_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
        
        left_vis = keypoints[left_wrist_idx][2]
        right_vis = keypoints[right_wrist_idx][2]
        left_z = z_coords[left_wrist_idx]
        right_z = z_coords[right_wrist_idx]

        left_valid = left_vis > visibility_threshold
        right_valid = right_vis > visibility_threshold

        best_side = None
        if left_valid and right_valid:
            # Nếu cả hai tay đều hợp lệ, chọn tay có cổ tay gần camera hơn
            best_side = 'LEFT' if left_z < right_z else 'RIGHT'
        elif left_valid:
            best_side = 'LEFT'
        elif right_valid:
            best_side = 'RIGHT'

        if best_side:
            # Gọi staticmethod khác bằng tên Class
            coords = HumanDetection._get_limb_coords(best_side, 'arm', keypoints, z_coords)
            return best_side, coords
        
        return None, None

    @staticmethod
    def select_best_leg(keypoints: np.ndarray, z_coords: np.ndarray, visibility_threshold: float = 0.8):
        """
        [STATICMETHOD] Chọn chân tốt nhất dựa trên visibility và khoảng cách Z.
        """
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
        right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

        left_vis = keypoints[left_ankle_idx][2]
        right_vis = keypoints[right_ankle_idx][2]
        left_z = z_coords[left_ankle_idx]
        right_z = z_coords[right_ankle_idx]

        left_valid = left_vis > visibility_threshold
        right_valid = right_vis > visibility_threshold
        
        best_side = None
        if left_valid and right_valid:
            # Nếu cả hai chân đều hợp lệ, chọn chân có mắt cá gần camera hơn
            best_side = 'LEFT' if left_z < right_z else 'RIGHT'
        elif left_valid:
            best_side = 'LEFT'
        elif right_valid:
            best_side = 'RIGHT'

        if best_side:
            # Gọi staticmethod khác bằng tên Class
            coords = HumanDetection._get_limb_coords(best_side, 'leg', keypoints, z_coords)
            return best_side, coords

        return None, None

    @staticmethod
    def get_torso_box(keypoints: np.ndarray, visibility_threshold: float = 0.8):
        """
        [STATICMETHOD] Tính toán bounding box cho phần thân trên (torso)
        dựa trên vị trí của vai và hông.
        """
        torso_indices = [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value
        ]
        torso_points = []
        for idx in torso_indices:
            # Đảm bảo chỉ số không vượt quá giới hạn của mảng keypoints
            if idx < len(keypoints) and keypoints[idx][2] > visibility_threshold:
                torso_points.append(keypoints[idx][:2])
        
        # Cần ít nhất 3 điểm để xác định một vùng đáng tin cậy
        if len(torso_points) < 3:
            return None

        torso_points = np.array(torso_points, dtype=np.int32)
        x1 = np.min(torso_points[:, 0])
        y1 = np.min(torso_points[:, 1])
        x2 = np.max(torso_points[:, 0])
        y2 = np.max(torso_points[:, 1])

        return x1, y1, x2, y2

    def draw_results(self, image: np.ndarray, min_conf: float = 0.5):
        """Vẽ kết quả phát hiện (bounding box, skeleton) lên ảnh."""
        if self.last_results is None:
            return image
            
        annotated_image = image.copy()
        keypoints_data, boxes_data = self.last_results
        
        for i, (kpts, box) in enumerate(zip(keypoints_data, boxes_data)):
            x1, y1, x2, y2, conf = box
            
            # Vẽ bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {i+1} ({conf:.2f})"
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Vẽ các điểm keypoint và nối chúng thành skeleton
            points = {}
            for j, (x, y, visibility) in enumerate(kpts):
                if visibility > min_conf:
                    points[j] = (int(x), int(y))
                    cv2.circle(annotated_image, points[j], 3, (0, 0, 255), -1)
            
            for start_idx, end_idx in MEDIAPIPE_EDGES:
                if start_idx in points and end_idx in points:
                    cv2.line(annotated_image, points[start_idx], points[end_idx], (255, 255, 0), 1)
                    
        return annotated_image

# Khối chạy chính để kiểm tra
if __name__ == "__main__":
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        print(f"Vui lòng tải model '{model_path}' và đặt vào cùng thư mục.")
        exit()

    detector = HumanDetection(pose_model=model_path)
    
    video_path = "path/to/your/video.mp4" # << THAY ĐỔI ĐƯỜNG DẪN NÀY
    if not os.path.exists(video_path):
        print(f"Không tìm thấy file video: {video_path}. Vui lòng kiểm tra lại đường dẫn.")
        exit()
        
    source = cv2.VideoCapture(video_path) 

    while True:
        ret, frame = source.read()
        if not ret:
            print("Kết thúc video hoặc không thể đọc frame.")
            break
        
        keypoints_data, boxes_data, z_data = detector.run_detection(frame)
        
        if len(keypoints_data) > 0:
            print("\n" + "="*40)
            
            # Lặp qua từng người để tìm chi tốt nhất
            for i in range(len(boxes_data)):
                person_kpts = keypoints_data[i]
                person_z = z_data[i]
                box = boxes_data[i]
                x1, y1, _, _ = map(int, box[:4])
                
                print(f"--- Người {i+1} ---")
                
                # *** GỌI HÀM CHỌN TAY (dưới dạng staticmethod) ***
                best_arm_side, arm_coords = HumanDetection.select_best_arm(person_kpts, person_z)
                if best_arm_side:
                    print(f"  💪 Cánh tay tốt nhất: {best_arm_side}")
                    cv2.putText(frame, f"ARM: {best_arm_side}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # *** GỌI HÀM CHỌN CHÂN (dưới dạng staticmethod) ***
                best_leg_side, leg_coords = HumanDetection.select_best_leg(person_kpts, person_z)
                if best_leg_side:
                    print(f"  🦵 Chân tốt nhất: {best_leg_side}")
                    cv2.putText(frame, f"LEG: {best_leg_side}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        annotated_frame = detector.draw_results(frame)
        cv2.imshow("Hybrid Pose Detection - Limb Selection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()