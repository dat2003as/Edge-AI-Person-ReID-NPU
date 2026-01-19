"""
ClothingClassifier - Phiên bản CIX NPU (FIXED)
- Sử dụng NOE Engine đúng chuẩn theo test script
- Output format: YOLOv11 (cx, cy, w, h, class_scores...)
- Shape: (1, 10, 8400) cho 6 classes
- Tối ưu cho Orange Pi 6 Plus (RK3588 NPU)
"""

import cv2
import numpy as np
import csv
import os
from typing import Optional, List, Dict, Any
from collections import Counter

from utils.logging_python_orangepi import get_logger
from utils.NOE_Engine import EngineInfer

logger = get_logger(__name__)


class ClothingClassifier:
    def __init__(
        self,
        skin_csv_path: str,
        cix_model_path: str,
        min_conf: float = 0.35,  # Đồng bộ với test script
        sleeve_color_sim_threshold: float = 20.0,
        pants_color_sim_threshold: float = 50.0,
        dominant_lab_threshold: float = 18.0,
        input_size: tuple = (640, 640),
        iou_threshold: float = 0.35,
    ):
        self.min_conf = min_conf
        self.sleeve_threshold = sleeve_color_sim_threshold
        self.pants_threshold = pants_color_sim_threshold
        self.lab_threshold = dominant_lab_threshold
        self.input_size = input_size
        self.iou_threshold = iou_threshold

        self.cix_model_path = cix_model_path
        self.model = None
        self.model_load_failed = False

        self.skin_tone_palette = self._load_skin_tone_palette(skin_csv_path)

        self.SKIN_LOWER = np.array([0, 133, 77], dtype=np.uint8)
        self.SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)
        self.MIN_SKIN_PIXELS = 50

        # Class names PHẢI KHỚP với model training
        self.class_names = [
            "Ao ngan",    # 0
            "Ao dai",     # 1
            "Quan ngan",  # 2
            "Quan dai",   # 3
            "Vay ngan",   # 4
            "Vay dai"     # 5
        ]

        logger.info(f"[ClothingClassifier] Khởi tạo CIX | min_conf={min_conf} | model={cix_model_path}")
        self._load_cix_model()

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

    def _load_cix_model(self):
        """Load CIX model qua NOE Engine"""
        if self.model_load_failed:
            return

        try:
            if not os.path.exists(self.cix_model_path):
                raise FileNotFoundError(f"Không tìm thấy CIX model: {self.cix_model_path}")

            logger.info(f"[CIX] Đang load model: {self.cix_model_path}")
            self.model = EngineInfer(self.cix_model_path)
            logger.info("[CIX] Model load thành công")

            # Smoke test
            logger.info("[CIX] Chạy smoke test...")
            dummy = np.zeros((1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
            _ = self.model.forward(dummy)
            logger.info("[CIX] Smoke test OK")

        except Exception as e:
            logger.error(f"[CIX] Load thất bại: {e}", exc_info=True)
            self.model = None
            self.model_load_failed = True

    def _letterbox(self, img: np.ndarray, size: tuple = (640, 640)) -> np.ndarray:
        """
        Resize image với letterbox padding (giống test script)
        """
        h, w = img.shape[:2]
        r = min(size[0]/h, size[1]/w)
        new_h, new_w = int(h*r), int(w*r)
        img = cv2.resize(img, (new_w, new_h))
        
        dh, dw = size[0]-new_h, size[1]-new_w
        top, left = dh//2, dw//2
        bottom, right = dh-top, dw-left
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        🔥 FIXED: Preprocess đúng chuẩn theo test script
        1. Letterbox resize
        2. BGR -> RGB
        3. Transpose HWC -> CHW
        4. Add batch dimension
        5. Normalize [0-255] -> [0-1] float32
        """
        # 1. Resize (Letterbox)
        img_resized = self._letterbox(image, self.input_size)
        
        # 2. BGR -> RGB (Quan trọng vì YOLO train RGB)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Transpose về NCHW (1, 3, 640, 640)
        img_input = img_rgb.transpose(2, 0, 1)[np.newaxis, ...]
        
        # 4. Normalize và convert sang float32
        img_input = img_input.astype(np.float32) / 255.0
        
        # 5. Đảm bảo contiguous array
        return np.ascontiguousarray(img_input, dtype=np.float32)

    def _postprocess_detections(self, outputs: List[np.ndarray]) -> tuple:
        """
        🔥 FIXED: Post-process đúng format YOLOv11
        Output shape: (1, 10, 8400)
        - 4 channels đầu: cx, cy, w, h
        - 6 channels sau: class scores (6 classes)
        """
        try:
            # Lấy output đầu tiên (flat array từ NOE)
            output_flat = outputs[0]
            
            # Reshape về (1, Channels, Anchors)
            num_anchors = 8400
            num_channels = output_flat.size // num_anchors  # Ví dụ: 10 (4 box + 6 class)
            
            pred = output_flat.reshape(1, num_channels, num_anchors)
            pred = pred[0].transpose(1, 0)  # Shape: (8400, 10)
            
            # Tách box coordinates và class scores
            boxes_xywh = pred[:, :4]  # (8400, 4) - cx, cy, w, h
            class_logits = pred[:, 4:]  # (8400, 6) - class scores
            
            # Tính class ID và confidence
            class_ids = np.argmax(class_logits, axis=1)
            confidences = np.max(class_logits, axis=1)
            
            # 🔥 BỘ LỌC 1: Confidence threshold
            mask_conf = confidences > self.min_conf
            
            # 🔥 BỘ LỌC 2: Lọc ghost boxes (box quá to)
            w = boxes_xywh[:, 2]
            h = boxes_xywh[:, 3]
            area = w * h
            img_area = 640 * 640
            mask_size = area < (img_area * 0.90)  # Box < 90% ảnh
            
            # Kết hợp 2 điều kiện
            mask = mask_conf & mask_size
            
            boxes_xywh = boxes_xywh[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            
            if len(confidences) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # 🔥 Convert xywh (center) -> xyxy (top-left, bottom-right)
            boxes_xyxy = np.copy(boxes_xywh)
            boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
            
            return boxes_xyxy, confidences, class_ids
            
        except Exception as e:
            logger.error(f"[CIX] Lỗi postprocess: {e}", exc_info=True)
            return np.array([]), np.array([]), np.array([])

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """
        Non-maximum suppression (đúng theo test script)
        """
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= self.iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

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

        valid = [c for c in colors_bgr if 40 < sum(c) < 680]
        if len(valid) < 3:
            return valid[0] if valid else None

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

    def _save_debug_image(self, image: np.ndarray, boxes: np.ndarray, 
                          scores: np.ndarray, class_ids: np.ndarray, 
                          save_path: str = "debug_clothing_detection.jpg"):
        """
        Vẽ bounding boxes lên ảnh để debug
        - Màu đỏ: Detection boxes
        - Label: Class name + confidence
        """
        debug_img = image.copy()
        
        for box, score, cls_id in zip(boxes, scores, class_ids):
            cls_id = int(cls_id)
            x1, y1, x2, y2 = map(int, box)
            
            # Vẽ box màu đỏ
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Vẽ label
            label = f"{self.class_names[cls_id]}: {score:.2f}"
            
            # Background cho text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(debug_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)
            
            # Text trắng
            cv2.putText(debug_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Lưu ảnh
        cv2.imwrite(save_path, debug_img)
        logger.info(f"✅ [DEBUG] Đã lưu ảnh debug tại: {save_path}")
        logger.info(f"📊 [DEBUG] Số boxes: {len(boxes)}")
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
            logger.info(f"   Box {i+1}: {self.class_names[int(cls_id)]} | conf={score:.3f} | bbox={box.astype(int).tolist()}")

    def classify(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🔍 Bắt đầu phân loại quần áo + màu sắc (CIX NPU)")

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

        def safe_get_list(key: str) -> List[Dict]:
            val = regional.get(key)
            if val is None or not isinstance(val, list):
                return []
            return val

        brachium_list = safe_get_list("brachium_colors")
        thigh_list = safe_get_list("thigh_colors")
        forearm_list = safe_get_list("forearm_colors")

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

        # ─── 3. CIX NPU phân loại kiểu dáng ────────────────────────────────
        yolo_result = {"sleeve_type": "Chưa xác định", "pants_type": "Chưa xác định"}
        max_conf = 0.0

        if self.model and not self.model_load_failed:
            try:
                # Preprocess
                input_tensor = self._preprocess_image(image)
                
                # Inference
                logger.debug("[CIX] Đang chạy inference...")
                outputs = self.model.forward(input_tensor)
                
                # Postprocess
                boxes, scores, class_ids = self._postprocess_detections(outputs)
                
                # NMS
                if len(boxes) > 0:
                    keep = self._nms(boxes, scores)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    class_ids = class_ids[keep]
                
                logger.info(f"[CIX] Tìm thấy {len(boxes)} detections sau NMS")

                # 🔥 LƯU ẢNH DEBUG (chỉ lưu khi có detection)
                # if len(boxes) > 0:
                #     try:
                #         # Tạo thư mục debug nếu chưa có
                #         debug_dir = "debug_clothing"
                #         os.makedirs(debug_dir, exist_ok=True)
                        
                #         # Tạo tên file unique theo timestamp
                #         import time
                #         timestamp = int(time.time() * 1000)
                #         debug_path = os.path.join(debug_dir, f"clothing_detect_{timestamp}.jpg")
                        
                #         self._save_debug_image(image, boxes, scores, class_ids, debug_path)
                #     except Exception as e:
                #         logger.error(f"Lỗi khi lưu debug image: {e}")

                sleeve_type = pants_type = "Chưa xác định"

                for box, score, cls_id in zip(boxes, scores, class_ids):
                    cls_id = int(cls_id)
                    name = self.class_names[cls_id].lower()
                    
                    if score > max_conf:
                        max_conf = score

                    # Phân loại kiểu dáng
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

                logger.info(f"[CIX] Kết quả: {sleeve_type} | {pants_type} | conf={max_conf:.3f}")

            except Exception as e:
                logger.error(f"[CIX] Lỗi inference: {e}", exc_info=True)

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
        }

    def __del__(self):
        """Cleanup khi object bị destroy"""
        if self.model is not None:
            try:
                self.model.clean()
                logger.info("[CIX] Model cleaned up")
            except:
                pass