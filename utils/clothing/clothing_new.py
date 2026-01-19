# utils/clothing_new.py
"""
ClothingClassifier - Phiên bản hoàn chỉnh
- Sử dụng YOLO để phân loại kiểu dáng: áo ngắn/dài, quần/váy ngắn/dài
- Trích xuất màu chủ đạo từ vùng pose (brachium → áo, thigh → quần/váy)
- Ưu tiên skin tone từ forearm nếu có, fallback full frame
- Tối ưu cho Orange Pi 6 Plus (RK3588 NPU/CPU)
"""

import cv2
import numpy as np
import csv
import os
from typing import Optional, List, Dict, Any
from collections import Counter

from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)

from ultralytics import YOLO


class ClothingClassifier:
    def __init__(
        self,
        skin_csv_path: str,
        yolo_model_path: str,
        min_conf: float = 0.75,
        sleeve_color_sim_threshold: float = 20.0,
        pants_color_sim_threshold: float = 50.0,
        dominant_lab_threshold: float = 18.0,
    ):
        self.min_conf = min_conf
        self.sleeve_threshold = sleeve_color_sim_threshold
        self.pants_threshold = pants_color_sim_threshold
        self.lab_threshold = dominant_lab_threshold

        self.yolo_model_path = yolo_model_path
        self.model = None
        self.model_load_failed = False

        self.skin_tone_palette = self._load_skin_tone_palette(skin_csv_path)

        self.SKIN_LOWER = np.array([0, 133, 77], dtype=np.uint8)
        self.SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)
        self.MIN_SKIN_PIXELS = 50

        logger.info(f"[ClothingClassifier] Khởi tạo | min_conf={min_conf} | model={yolo_model_path}")
        self._load_yolo_model()

    def _load_skin_tone_palette(self, csv_path: str) -> List[Dict]:
        palette = []
        try:
            with open(csv_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        bgr = (int(row['b']), int(row['g']), int(row['r']))
                        palette.append({'id': int(row['id']), 'bgr': bgr})
                    except:
                        continue
        except Exception as e:
            logger.warning(f"Không đọc được skin palette CSV: {e}")

        if not palette:
            logger.warning("Dùng palette mặc định")
            palette = [
                {'id': 1, 'bgr': (145, 169, 210)},
                {'id': 2, 'bgr': (130, 175, 225)},
                {'id': 3, 'bgr': (115, 155, 200)},
            ]
        return palette

    def _load_yolo_model(self):
        if self.model_load_failed:
            return

        try:
            if not os.path.exists(self.yolo_model_path):
                raise FileNotFoundError(f"Không tìm thấy model: {self.yolo_model_path}")

            logger.info(f"Đang load YOLO clothing model: {self.yolo_model_path}")
            self.model = YOLO(self.yolo_model_path)
            logger.info("[YOLO] Model load thành công")

            # Smoke test
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, verbose=False)
            logger.info("[YOLO] Smoke test OK")

        except Exception as e:
            logger.error(f"[YOLO] Load thất bại: {e}", exc_info=True)
            self.model = None
            self.model_load_failed = True

    def _is_skin_color(self, bgr: List[int]) -> bool:
        if not bgr:
            return False
        ycrcb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]
        return np.all((self.SKIN_LOWER <= ycrcb) & (ycrcb <= self.SKIN_UPPER))

    def _extract_skin_mean(self, frame: np.ndarray) -> Optional[List[int]]:
        if frame is None or frame.size == 0:
            return None

        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, self.SKIN_LOWER, self.SKIN_UPPER)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)

            pixels = frame[mask > 0]
            if len(pixels) < self.MIN_SKIN_PIXELS:
                return None

            return np.mean(pixels, axis=0).astype(int).tolist()
        except:
            return None

    def _find_closest_skin_tone(self, bgr: List[int]) -> tuple:
        if not bgr:
            return None, None

        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)
        best, min_dist = None, float('inf')

        for tone in self.skin_tone_palette:
            tlab = cv2.cvtColor(np.uint8([[tone['bgr']]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)
            dist = np.linalg.norm(lab - tlab)
            if dist < min_dist:
                min_dist, best = dist, tone

        return best['id'] if best else None, best['bgr'] if best else None

    def _dominant_color_lab(self, colors_bgr: List[List[int]]) -> Optional[List[int]]:
        if not colors_bgr or len(colors_bgr) < 3:
            return colors_bgr[0] if colors_bgr else None

        # Lọc màu hợp lý
        valid = [c for c in colors_bgr if 40 < sum(c) < 680]
        if len(valid) < 3:
            return valid[0] if valid else None

        # Nhóm màu trong LAB
        clusters = []
        for c in valid:
            lab = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2LAB)[0][0]
            clusters.append((
                round(lab[0] / 10) * 10,
                round(lab[1] / 8) * 8,
                round(lab[2] / 8) * 8,
            ))

        most_common = Counter(clusters).most_common(1)
        if not most_common:
            return None

        lab_rep = np.uint8([[most_common[0][0]]])
        return cv2.cvtColor(lab_rep, cv2.COLOR_LAB2BGR)[0][0].tolist()

    def classify(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🔍 Bắt đầu phân loại quần áo + màu sắc")

        image = analysis_data.get('image')
        if image is None or image.size == 0:
            logger.error("Không có ảnh đầu vào")
            return self._default_result()

        h, w = image.shape[:2]
        logger.debug(f"Kích thước ảnh: {w}x{h}")

        # ─── 1. Phát hiện da ───────────────────────────────────────────────
        skin_bgr = self._extract_skin_mean(image)
        skin_id, skin_palette = self._find_closest_skin_tone(skin_bgr)

        # ─── 2. Trích xuất màu từ vùng pose ────────────────────────────────
        upper_colors = []
        lower_colors = []
        forearm_skin_colors = []

        regional = analysis_data.get("regional_analysis", {}) or {}

        # Safe get list
        def safe_get_list(key: str) -> List[Dict]:
            val = regional.get(key)
            if val is None or not isinstance(val, list):
                logger.debug(f"[Clothing] {key} là None hoặc không phải list → fallback []")
                return []
            return val

        brachium_list = safe_get_list("brachium_colors")
        thigh_list = safe_get_list("thigh_colors")
        forearm_list = safe_get_list("forearm_colors")

        # DEBUG LOG SAU KHI GÁN (fix UnboundLocalError)
        logger.info(f"[Clothing DEBUG] brachium_list len: {len(brachium_list)}")
        logger.info(f"[Clothing DEBUG] thigh_list len: {len(thigh_list)}")
        logger.info(f"[Clothing DEBUG] forearm_list len: {len(forearm_list)}")

        # Brachium → màu áo
        for item in brachium_list:
            if not isinstance(item, dict):
                continue
            bgr = item.get("bgr")
            if not isinstance(bgr, list) or len(bgr) != 3:
                continue
            perc = item.get("percentage", 0)
            if perc > 5:
                upper_colors.append(bgr)

        # Thigh → màu quần/váy
        for item in thigh_list:
            if not isinstance(item, dict):
                continue
            bgr = item.get("bgr")
            if not isinstance(bgr, list) or len(bgr) != 3:
                continue
            perc = item.get("percentage", 0)
            if perc > 5:
                lower_colors.append(bgr)

        # Forearm → skin tone
        for item in forearm_list:
            if not isinstance(item, dict):
                continue
            bgr = item.get("bgr")
            if not isinstance(bgr, list) or len(bgr) != 3:
                continue
            if self._is_skin_color(bgr):
                forearm_skin_colors.append(bgr)

        # Ưu tiên skin từ forearm
        if forearm_skin_colors:
            skin_bgr = np.mean(forearm_skin_colors, axis=0).astype(int).tolist()
            skin_id, skin_palette = self._find_closest_skin_tone(skin_bgr)

        upper_dom = self._dominant_color_lab(upper_colors)
        lower_dom = self._dominant_color_lab(lower_colors)

        # ─── 3. YOLO phân loại kiểu dáng ───────────────────────────────────
        yolo_result = {"sleeve_type": "Chưa xác định", "pants_type": "Chưa xác định"}
        # debug_yolo = {"total_detections": 0, "valid_detections": 0}

        if self.model and not self.model_load_failed:
            try:
                input_img = image
                max_dim = max(h, w)
                if max_dim > 640:
                    scale = 640 / max_dim
                    input_img = cv2.resize(image, (int(w * scale), int(h * scale)))

                detections = []
                for conf_lvl in [self.min_conf, 0.25, 0.15]:
                    results = self.model.predict(
                        input_img,
                        conf=conf_lvl,
                        iou=0.5,
                        verbose=False,
                        device='cpu'
                    )

                    for r in results:
                        if r.boxes is None or len(r.boxes) == 0:
                            continue
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf_val = float(box.conf[0])
                            class_names = [
                                "Ao ngan", "Ao dai",
                                "Quan ngan", "Quan dai",
                                "Vay ngan", "Vay dai"
                            ]
                            if cls_id < len(class_names):
                                detections.append({
                                    "class_name": class_names[cls_id],
                                    "confidence": conf_val,
                                    "bbox": box.xyxy[0].tolist()
                                })

                    if detections:
                        break

                valid_dets = [d for d in detections if d['confidence'] >= 0.20]
                # debug_yolo = {
                #     "total_detections": len(detections),
                #     "valid_detections": len(valid_dets),
                #     "all_detections": detections[:4]  # giới hạn log
                # }

                sleeve_type = pants_type = "Chưa xác định"
                max_conf = 0.0

                for det in valid_dets:
                    name = det['class_name'].lower()
                    conf = det['confidence']
                    if conf > max_conf:
                        max_conf = conf

                    if "ngan" in name and "ao" in name:
                        sleeve_type = "Ao ngan"
                    elif "dai" in name and "ao" in name:
                        sleeve_type = "Ao dai"

                    if "ngan" in name and ("quan" in name or "vay" in name):
                        pants_type = "Quan ngan" if "quan" in name else "Vay ngan"
                    elif "dai" in name and ("quan" in name or "vay" in name):
                        pants_type = "Quan dai" if "quan" in name else "Vay dai"

                yolo_result = {
                    "sleeve_type": sleeve_type,
                    "pants_type": pants_type,
                }

                logger.info(f"[YOLO] Kết quả: {sleeve_type} | {pants_type} | conf={max_conf:.3f}")

            except Exception as e:
                logger.error(f"[YOLO] Lỗi inference: {e}", exc_info=True)

        # ─── 4. Kết quả cuối ───────────────────────────────────────────────
        result = {
            "upper_type": yolo_result["sleeve_type"],
            "lower_type": yolo_result["pants_type"],

            "upper_color": upper_dom,
            "lower_color": lower_dom,
            "skin_tone_bgr": skin_palette,
            "skin_tone_id": skin_id,

            "raw_colors": {
                "brachium": regional.get("brachium_colors"),
                "forearm": regional.get("forearm_colors"),
                "thigh": regional.get("thigh_colors"),
                "shin": regional.get("shin_colors"),
            },

            "classification_details": {
                "sleeve": {"type": yolo_result["sleeve_type"], "confidence": max_conf},
                "pants": {"type": yolo_result["pants_type"], "confidence": max_conf},
            },
            "skin_flags": {"forearm_skin_detected": len(forearm_skin_colors) > 0}
        }

        # logger.info(
        #     f"[CLOTHING RESULT] "
        #     f"Áo: {result['upper_type']:<10} | Quần/Váy: {result['lower_type']:<10} | "
        #     f"Màu áo: {result['upper_color']} | Màu dưới: {result['lower_color']} | "
        #     f"Da: {result['skin_tone_bgr']}"
        # )

        return result

    def _default_result(self) -> Dict[str, Any]:
        return {
            "sleeve_type": "Chưa xác định",
            "pants_type": "Chưa xác định",
            "upper_type": "Chưa xác định",
            "lower_type": "Chưa xác định",
            "upper_color": None,
            "lower_color": None,
            "skin_tone_bgr": None,
            "skin_tone_id": None,
            "raw_colors": {},
            "classification_details": {},
            "skin_flags": {"forearm_skin_detected": False},
            "debug_info": {"color_pixels": {"brachium_count": 0, "thigh_count": 0, "forearm_skin_count": 0}}
        }