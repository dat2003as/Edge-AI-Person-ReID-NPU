# utils/pose_color_new1.py
import asyncio
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple, Dict, Any

# Giả định các import này đúng với cấu trúc dự án của bạn
from utils.clothing.clothing_classifier_by_color_new import ClothingClassifier
from utils.logging_python_orangepi import get_logger
from utils.detectors.mediapipe_pose import HumanDetection # << IMPORT QUAN TRỌNG

logger = get_logger(__name__)

class PoseColorAnalyzer:
    """
    [PHIÊN BẢN TÁI CẤU TRÚC]
    - Sử dụng các hàm tiện ích từ HumanDetection để xác định vùng cơ thể.
    - Chỉ phân tích màu trên các chi "tốt nhất" (gần camera nhất, rõ nhất).
    - Tối ưu hóa logic, loại bỏ các hằng số và các hàm trích xuất dư thừa.
    """
    
    # Đã loại bỏ các hằng số định nghĩa keypoint cũ

    def __init__(self, line_thickness: int = 30, k_clusters: int = 3):
        self.line_thickness = line_thickness
        self.k_clusters = k_clusters
        self.MIN_PIXELS_FOR_COLOR = 50
        self.MIN_PERCENTAGE = 7.0
        self.MERGE_THRESHOLD = 50.0
        self.MONOCHROMATIC_THRESHOLD = 80.0
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0

    def _is_valid_point(self, point) -> bool:
        """Kiểm tra điểm keypoint có hợp lệ không."""
        return point is not None and len(point) >= 2 and point[0] != 0 and point[1] != 0

    def _get_pixels_from_polygon(self, image: np.ndarray, points: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """Trích xuất pixels từ vùng đa giác (dùng cho torso)."""
        if len(points) < 3:
            return None
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 255)
            pixels = image[mask == 255]
            return pixels if len(pixels) >= self.MIN_PIXELS_FOR_COLOR else None
        except Exception as e:
            logger.debug(f"Lỗi trích xuất pixels từ polygon: {e}")
            return None

    def _get_pixels_from_line(self, image: np.ndarray, start_point: tuple, end_point: tuple) -> Optional[np.ndarray]:
        """Trích xuất pixels từ một đường thẳng có độ dày (dùng cho tay, chân)."""
        try:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.line(mask, start_point, end_point, 255, self.line_thickness)
            pixels = image[mask == 255]
            return pixels if len(pixels) >= 10 else None
        except Exception:
            return None

    def _calculate_color_distance_lab(self, color1_bgr: np.ndarray, color2_bgr: np.ndarray) -> float:
        """Tính khoảng cách màu trong không gian LAB để so sánh màu sắc chính xác hơn."""
        try:
            color1_lab = cv2.cvtColor(np.uint8([[color1_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            color2_lab = cv2.cvtColor(np.uint8([[color2_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            return np.linalg.norm(color1_lab.astype(float) - color2_lab.astype(float))
        except:
            return float('inf')

    def _merge_similar_colors(self, colors: List[np.ndarray], percentages: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """Gộp các màu tương tự trong bảng màu để kết quả gọn hơn."""
        if len(colors) < 2:
            return colors, percentages
        merged_colors = colors.copy()
        merged_percentages = percentages.copy()
        i = 0
        while i < len(merged_colors):
            j = i + 1
            while j < len(merged_colors):
                distance = self._calculate_color_distance_lab(merged_colors[i], merged_colors[j])
                if distance < self.MERGE_THRESHOLD:
                    total_percentage = merged_percentages[i] + merged_percentages[j]
                    new_color = ((merged_colors[i] * merged_percentages[i] + merged_colors[j] * merged_percentages[j]) / total_percentage)
                    merged_colors[i] = new_color
                    merged_percentages[i] = total_percentage
                    merged_colors.pop(j)
                    merged_percentages.pop(j)
                    j -= 1
                j += 1
            i += 1
        return merged_colors, merged_percentages

    def analyze_colors_simple(self, pixels: np.ndarray) -> Optional[List[Dict]]:
        """Phân tích màu đơn giản - chỉ lấy màu trung bình."""
        if pixels is None or len(pixels) == 0:
            return None
        try:
            mean_color = np.mean(pixels, axis=0).astype(int)
            return [{"bgr": mean_color.tolist(), "percentage": 100.0}]
        except Exception as e:
            logger.debug(f"Lỗi phân tích màu đơn giản: {e}")
            return None

    def analyze_colors_advanced(self, pixels: np.ndarray) -> Optional[List[Dict]]:
        """Phân tích màu nâng cao bằng K-Means để trích xuất bảng màu."""
        if pixels is None or len(pixels) < self.k_clusters:
            return self.analyze_colors_simple(pixels)
        try:
            pixels_rgb = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_BGR2RGB)[0]
            k = max(2, min(self.k_clusters, len(pixels) // 20 if len(pixels) >= 40 else 2))
            
            kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
            kmeans.fit(pixels_rgb)
            
            total_pixels = len(kmeans.labels_)
            counts = np.bincount(kmeans.labels_)
            colors_rgb = kmeans.cluster_centers_
            colors_bgr = cv2.cvtColor(np.uint8([colors_rgb]), cv2.COLOR_RGB2BGR)[0]
            
            valid_colors, valid_percentages = [], []
            for i, count in enumerate(counts):
                percentage = (count / total_pixels) * 100
                if percentage >= self.MIN_PERCENTAGE:
                    valid_colors.append(colors_bgr[i])
                    valid_percentages.append(percentage)
            
            if not valid_colors:
                return self.analyze_colors_simple(pixels)
                
            merged_colors, merged_percentages = self._merge_similar_colors(valid_colors, valid_percentages)
            sorted_pairs = sorted(zip(merged_percentages, merged_colors), key=lambda x: x[0], reverse=True)
            
            if len(sorted_pairs) == 1 or sorted_pairs[0][0] > self.MONOCHROMATIC_THRESHOLD:
                return [{"bgr": sorted_pairs[0][1].astype(int).tolist(), "percentage": round(sorted_pairs[0][0], 2)}]
                
            return [{"bgr": color.astype(int).tolist(), "percentage": round(percentage, 2)} for percentage, color in sorted_pairs]
        except Exception as e:
            logger.debug(f"Lỗi phân tích màu nâng cao: {e}")
            return self.analyze_colors_simple(pixels)

    def _analyze_limb_segment(self, image: np.ndarray, p1: tuple, p2: tuple, is_forearm: bool) -> Optional[List[Dict]]:
        """Hàm helper để phân tích màu một đoạn chi (tay/chân)."""
        if not self._is_valid_point(p1) or not self._is_valid_point(p2):
            return None
        pixels = self._get_pixels_from_line(image, p1, p2)
        if pixels is None:
            return None
        
        # Cẳng tay ('forearm') cần phân tích đa màu để có thể phát hiện màu da
        if is_forearm:
            return self.analyze_colors_advanced(pixels)
        else: # Các bộ phận khác chỉ cần màu chủ đạo để phân loại quần/áo
            colors = self.analyze_colors_advanced(pixels)
            return [max(colors, key=lambda x: x['percentage'])] if colors else None

    def _update_fps(self):
        """Cập nhật tính toán FPS.""" 
        self.frame_count += 1
        current_time = asyncio.get_event_loop().time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1:
            self.fps = self.frame_count / elapsed_time
            logger.info(f"FPS POSE_COLOR: {self.fps:.2f}")
            self.start_time = current_time
            self.frame_count = 0


    async def process_and_classify(
        self, 
        image: np.ndarray, 
        keypoints: np.ndarray,
        classifier: ClothingClassifier, 
        kpts_z: Optional[np.ndarray] = None,
        external_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Pipeline xử lý phân tích màu quần áo (Đã vá lỗi NoneType).
        """
        # --- BƯỚC QUAN TRỌNG: VÁ LỖI CRASH ---
        # Kiểm tra ảnh đầu vào
        if image is None or image.size == 0:
            return None
        if keypoints is None:
            return None 
        if external_data is None: external_data = {}
 
        
        try:
            # BƯỚC 1: Xác định các vùng cơ thể
            # Vì đã kiểm tra keypoints is None ở trên, các hàm dưới này sẽ chạy an toàn
            torso_box = HumanDetection.get_torso_box(keypoints)
            _arm_side, arm_coords = HumanDetection.select_best_arm(keypoints, kpts_z)
            _leg_side, leg_coords = HumanDetection.select_best_leg(keypoints, kpts_z)

            # BƯỚC 2: Chuẩn bị các tác vụ (task) để chạy bất đồng bộ
            tasks = {}

            # --- Tác vụ cho Thân (Torso) ---
            if torso_box:
                x1, y1, x2, y2 = torso_box
                # Kiểm tra tọa độ hợp lệ
                if y2 > y1 and x2 > x1: 
                    torso_pixels = self._get_pixels_from_polygon(image, [(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
                    tasks['torso'] = asyncio.to_thread(self.analyze_colors_advanced, torso_pixels)

            # --- Tác vụ cho Cánh tay (Arm) ---
            if arm_coords:
                p_shoulder = tuple(map(int, (arm_coords['shoulder']['x'], arm_coords['shoulder']['y'])))
                p_elbow = tuple(map(int, (arm_coords['elbow']['x'], arm_coords['elbow']['y'])))
                p_wrist = tuple(map(int, (arm_coords['wrist']['x'], arm_coords['wrist']['y'])))
                tasks['brachium'] = asyncio.to_thread(self._analyze_limb_segment, image, p_shoulder, p_elbow, False)
                tasks['forearm'] = asyncio.to_thread(self._analyze_limb_segment, image, p_elbow, p_wrist, True)

            # --- Tác vụ cho Chân (Leg) ---
            if leg_coords:
                p_hip = tuple(map(int, (leg_coords['hip']['x'], leg_coords['hip']['y'])))
                p_knee = tuple(map(int, (leg_coords['knee']['x'], leg_coords['knee']['y'])))
                p_ankle = tuple(map(int, (leg_coords['ankle']['x'], leg_coords['ankle']['y'])))
                tasks['thigh'] = asyncio.to_thread(self._analyze_limb_segment, image, p_hip, p_knee, False)
                tasks['shin'] = asyncio.to_thread(self._analyze_limb_segment, image, p_knee, p_ankle, False)

            # BƯỚC 3: Thực thi đồng thời các tác vụ
            task_keys = list(tasks.keys())
            if not task_keys: 
                return None
            
            task_values = list(tasks.values())
            results = await asyncio.gather(*task_values, return_exceptions=True)

            # BƯỚC 4: Tổng hợp kết quả màu sắc
            regional_analysis = {
                "torso_colors": None, "brachium_colors": None, "forearm_colors": None,
                "thigh_colors": None, "shin_colors": None
            }
            for i, key in enumerate(task_keys):
                result = results[i]
                if not isinstance(result, Exception):
                    regional_analysis[f"{key}_colors"] = result

            # BƯỚC 5: Gọi classifier để phân loại loại áo/màu sắc
            data_for_classifier = {**external_data, "regional_analysis": regional_analysis}
            classification_result = await asyncio.to_thread(classifier.classify, data_for_classifier)

            self._update_fps()

            # Trả về kết quả cuối cùng
            return {
                "classification": classification_result, 
                "raw_color_data": regional_analysis,
                "processing_info": {
                    "k_clusters_default": self.k_clusters,
                    "best_arm_side": _arm_side,
                    "best_leg_side": _leg_side
                }
            }

        except Exception as e:
            logger.error(f"Lỗi pipeline phân tích màu: {e}", exc_info=True)
            return None

def create_analyzer(line_thickness: int = 30, k_clusters: int = 3) -> PoseColorAnalyzer:
    """Factory function để tạo analyzer với config mặc định."""
    return PoseColorAnalyzer(line_thickness=line_thickness, k_clusters=k_clusters)