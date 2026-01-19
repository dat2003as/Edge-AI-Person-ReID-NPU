# utils/clothing_classifier_by_color_new.py
import cv2
import numpy as np
import csv
from typing import Optional, List, Dict, Any

# Giả sử bạn có một logger đã được cấu hình
from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)

class ClothingClassifier:
    """
    [PHIÊN BẢN CẬP NHẬT - LOGIC MỚI]
    - Logic tìm da đáng tin cậy trước, sau đó mới dùng để phân loại áo.
    - Xử lý chính xác trường hợp áo dài tay 2 màu và các vấn đề do ánh sáng.
    - Cung cấp lý do phân loại rõ ràng hơn.
    """
    def __init__(
        self,
        skin_csv_path: str,
        # Tăng giá trị này (vd: 40-50) nếu hệ thống quá nhạy với bóng tối
        sleeve_color_similarity_threshold: float = 20.0,
        pants_color_similarity_threshold: float = 50.0
    ):
        self.sleeve_threshold = sleeve_color_similarity_threshold                                                                       
        self.pants_threshold = pants_color_similarity_threshold
        print(self.sleeve_threshold)
        # Load skin tone palette from CSV
        self.skin_tone_palette = self._load_skin_tone_palette(skin_csv_path)

        # Skin detection bounds in YCrCb color space
        self.SKIN_LOWER_BOUND = np.array([0, 133, 77])
        self.SKIN_UPPER_BOUND = np.array([255, 173, 127])
        self.MIN_SKIN_PIXELS = 50

    def _load_skin_tone_palette(self, csv_path: str) -> List[Dict]:
        palette = []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    try:
                        bgr_color = (int(row['b']), int(row['g']), int(row['r']))
                        palette.append({'id': int(row['id']), 'bgr': bgr_color})
                    except (ValueError, KeyError):
                        continue
        except Exception:
            pass
        # Fallback defaults
        if not palette:
            return [
                {'id': 1, 'bgr': (145, 169, 210)},
                {'id': 2, 'bgr': (130, 175, 225)},
                {'id': 3, 'bgr': (115, 155, 200)}
            ]
        return palette

    def _are_colors_similar(self, c1_bgr: List[int], c2_bgr: List[int], threshold: float) -> bool:
        if c1_bgr is None or c2_bgr is None:
            return False
        # Convert to LAB and measure distance
        lab1 = cv2.cvtColor(np.uint8([[c1_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        lab2 = cv2.cvtColor(np.uint8([[c2_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        dist = np.linalg.norm(lab1.astype(float) - lab2.astype(float))
        return dist < threshold

    def _calculate_color_confidence(self, c1_bgr: List[int], c2_bgr: List[int]) -> float:
        # Returns normalized confidence 0-1 based on LAB distance
        if c1_bgr is None or c2_bgr is None:
            return 0.0
        lab1 = cv2.cvtColor(np.uint8([[c1_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        lab2 = cv2.cvtColor(np.uint8([[c2_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        dist = np.linalg.norm(lab1.astype(float) - lab2.astype(float))
        max_dist = 100.0
        return max(0.0, 1.0 - (dist / max_dist))

    def _is_skin_color_ycrcb(self, bgr_color: List[int]) -> bool:
        if bgr_color is None:
            return False
        ycrcb = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2YCrCb)[0][0]
        return (
            self.SKIN_LOWER_BOUND[0] <= ycrcb[0] <= self.SKIN_UPPER_BOUND[0] and
            self.SKIN_LOWER_BOUND[1] <= ycrcb[1] <= self.SKIN_UPPER_BOUND[1] and
            self.SKIN_LOWER_BOUND[2] <= ycrcb[2] <= self.SKIN_UPPER_BOUND[2]
        )

    def _extract_skin_from_forearms(self, regional_data: Dict) -> Optional[List[int]]:
        forearm_colors = regional_data.get("forearm_colors") or []
        for info in forearm_colors:
            bgr = info.get("bgr")
            perc = info.get("percentage", 0)
            if perc >= 15.0 and self._is_skin_color_ycrcb(bgr):
                return bgr
        return None

    def _find_closest_skin_tone(self, bgr_color: List[int]) -> tuple:
        if bgr_color is None:
            return None, None
        lab_det = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2LAB)[0][0]
        best, best_dist = None, float('inf')
        for tone in self.skin_tone_palette:
            lab_pal = cv2.cvtColor(np.uint8([[tone['bgr']]]), cv2.COLOR_BGR2LAB)[0][0]
            d = np.linalg.norm(lab_det.astype(float) - lab_pal.astype(float))
            if d < best_dist:
                best_dist, best = d, tone
        return best['id'], best['bgr']

    # CẬP NHẬT: Hàm phân loại tay áo với logic mới
    def _classify_sleeve_type(
        self,
        data: Dict,
        reliable_skin_bgr: Optional[List[int]]
    ) -> Dict[str, Any]:
        bc = data.get("brachium_colors") or []
        fc = data.get("forearm_colors") or []
        if not bc or not fc:
            return {"sleeve_type": "CHUA XAC DINH", "confidence": 0.0, "reason": "Thiếu dữ liệu màu"}

        main_b = bc[0]["bgr"]
        main_f = fc[0]["bgr"]
        conf = self._calculate_color_confidence(main_b, main_f)

        # 1. Nếu màu cánh tay trên và cẳng tay giống nhau -> Áo dài tay
        if self._are_colors_similar(main_b, main_f, self.sleeve_threshold):
            return {"sleeve_type": "AO DAI TAY", "confidence": 1.0, "reason": "Màu cánh tay và cẳng tay giống nhau"}

        # 2. Nếu màu khác nhau, kiểm tra với "bằng chứng da" đã tìm được
        if reliable_skin_bgr:
            # Ngưỡng chặt hơn để so sánh da với da
            skin_similarity_threshold = 25.0
            if self._are_colors_similar(main_f, reliable_skin_bgr, skin_similarity_threshold):
                return {"sleeve_type": "AO NGAN TAY", "confidence": 1.0, "reason": "Màu cẳng tay khớp với màu da"}
            else:
                return {"sleeve_type": "AO DAI TAY", "confidence": conf, "reason": "Áo dài tay 2 màu"}

        # 3. Nếu màu khác nhau và không tìm thấy bằng chứng da nào
        return {"sleeve_type": "AO DAI TAY", "confidence": conf, "reason": "Màu khác nhau, không xác định được da"}

    def _classify_pants_type(self, data: Dict) -> Dict[str, Any]:
        tc, sc = data.get("thigh_colors") or [], data.get("shin_colors") or []
        if not tc or not sc:
            return {"pants_type": "CHUA XAC DINH", "confidence": 0.0, "reason": "Thiếu dữ liệu"}

        main_t, main_s = tc[0]["bgr"], sc[0]["bgr"]
        conf = self._calculate_color_confidence(main_t, main_s)
        
        # Giữ một ngưỡng tin cậy tối thiểu
        if conf < 0.2:
            return {"pants_type": "CHUA XAC DINH", "confidence": conf, "reason": f"Độ tin cậy thấp ({conf:.2f})"}

        if self._are_colors_similar(main_t, main_s, self.pants_threshold):
            return {"pants_type": "QUAN DAI", "confidence": conf, "reason": f"Màu đùi và ống quần giống nhau"}
        return {"pants_type": "QUAN NGAN", "confidence": conf, "reason": "Màu đùi và ống quần khác nhau"}

    # CẬP NHẬT: Hàm classify chính điều phối logic mới
    def classify(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        regional = analysis_data.get("regional_analysis", {})

        # --- LOGIC MỚI: TÌM DA ĐÁNG TIN CẬY TRƯỚC ---
        # (Trong tương lai có thể thêm logic tìm da ở mặt làm ưu tiên #1)
        reliable_skin_bgr = self._extract_skin_from_forearms(regional)
        skin_id, skin_bgr_from_palette = self._find_closest_skin_tone(reliable_skin_bgr)
        logger.info(f"skin_brg : {skin_bgr_from_palette}+++++++++++++++++++++++++++++++++")
        srgb = skin_bgr_from_palette
        # srgb = cv2.cvtColor(skin_bgr_from_palette, cv2.COLOR_BGR2RGB)
        if skin_bgr_from_palette:
            
            srgb= list(skin_bgr_from_palette)[::-1]  
            logger.info(f"skin_rgb : {srgb}+++++++++++++++++++++++++++++++++")
        # --- PHÂN LOẠI DỰA TRÊN THÔNG TIN ĐÃ CÓ ---
        sleeve = self._classify_sleeve_type(regional, reliable_skin_bgr)
        pants = self._classify_pants_type(regional)

        return {
            "sleeve_type": sleeve["sleeve_type"],
            "pants_type": pants["pants_type"], 
            "skin_tone_bgr": skin_bgr_from_palette,
            "skin_tone_id": skin_id,
            "classification_details": {"sleeve": sleeve, "pants": pants},
            "skin_flags": {"forearm_skin_detected": reliable_skin_bgr is not None},
            "raw_colors": {
                "brachium": regional.get("brachium_colors"),
                "forearm": regional.get("forearm_colors"),
                "thigh": regional.get("thigh_colors"), 
                "shin": regional.get("shin_colors") 
            }
        }