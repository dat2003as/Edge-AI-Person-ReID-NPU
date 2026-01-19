import cv2
import config
import numpy as np
import logging
import time
import requests
from utils.detectors.cut_body_part import MEDIAPIPE_EDGES
from PIL import Image, ImageDraw, ImageFont
import os
from utils.open_cua_mqtt import open_door
from utils.send_option_video import send_gender_to_server
logger = logging.getLogger(__name__)

# ============================================================
# METADATA CACHE - Reduce DB queries for confirmed persons
# ============================================================
class MetadataCache:
    """
    LRU cache with TTL for confirmed person metadata.
    Reduces DB queries by caching stable data for 30 seconds.
    """
    
    def __init__(self, ttl_seconds=30, max_size=100):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache = {}  # {person_id: {'data': metadata, 'timestamp': time.time()}}
    
    def get(self, person_id, db_manager):
        """
        Get metadata from cache or DB.
        
        Args:
            person_id: Person ID to lookup
            db_manager: VectorDatabase_Manager instance
            
        Returns:
            Metadata dict or None
        """
        current_time = time.time()
        
        # Check cache first
        if person_id in self.cache:
            cached_entry = self.cache[person_id]
            age = current_time - cached_entry['timestamp']
            
            # Cache hit - still valid
            if age < self.ttl_seconds:
                logger.debug(f"[CACHE HIT] {person_id} (age: {age:.1f}s)")
                return cached_entry['data']
            else:
                # Expired - remove
                logger.debug(f"[CACHE EXPIRED] {person_id} (age: {age:.1f}s)")
                del self.cache[person_id]
        
        # Cache miss - query DB
        logger.debug(f"[CACHE MISS] {person_id} - querying DB")
        try:
            db_meta = db_manager.get_metadata(person_id)
            
            # Only cache if has CCCD data (confirmed persons)
            if db_meta and db_meta.get('cccd_matched'):
                # Cleanup old entries if cache is full
                if len(self.cache) >= self.max_size:
                    self._evict_oldest()
                
                # Store in cache
                self.cache[person_id] = {
                    'data': db_meta,
                    'timestamp': current_time
                }
                logger.debug(f"[CACHE STORE] {person_id} - cached for {self.ttl_seconds}s")
            
            return db_meta
        except Exception as e:
            logger.error(f"[CACHE ERROR] {person_id}: {e}")
            return None
    
    def _evict_oldest(self):
        """Remove oldest entry from cache"""
        if not self.cache:
            return
        
        oldest_id = min(self.cache.items(), key=lambda x: x[1]['timestamp'])[0]
        del self.cache[oldest_id]
        logger.debug(f"[CACHE EVICT] Removed {oldest_id}")
    
    def invalidate(self, person_id):
        """Manually invalidate cache entry (e.g., when status changes)"""
        if person_id in self.cache:
            del self.cache[person_id]
            logger.debug(f"[CACHE INVALIDATE] {person_id}")
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        logger.debug("[CACHE CLEAR] All entries removed")

# Global cache instance
metadata_cache = MetadataCache(ttl_seconds=30, max_size=100)

# Door opening cooldown mechanism
last_door_open_time = 0.0
DOOR_COOLDOWN_SECONDS = 8.0

# Video server configuration
last_video_request_time = 0.0
VIDEO_COOLDOWN_SECONDS = 15.0

# Load Vietnamese font
FONT_PATH = "./font/arial.ttf"
VIETNAMESE_FONT_SIZE = 32  # Tăng từ 20 -> 32 để chữ to hơn
VIETNAMESE_FONT = None

try:
    if os.path.exists(FONT_PATH):
        VIETNAMESE_FONT = ImageFont.truetype(FONT_PATH, VIETNAMESE_FONT_SIZE)
    else:
        logger.warning(f"Font file not found: {FONT_PATH}, will use default cv2 font for Vietnamese text")
except Exception as e:
    logger.warning(f"Failed to load Vietnamese font: {e}, will use default cv2 font")

def put_vietnamese_text(frame, text, pos, font_scale=0.5, color=(255, 255, 255), thickness=1):
    """
    Vẽ text tiếng Việt sử dụng PIL để hỗ trợ Unicode
    
    Args:
        frame: OpenCV frame (BGR)
        text: Text cần vẽ (có thể tiếng Việt)
        pos: Tuple (x, y)
        font_scale: Font scale (dùng cho cv2 nếu PIL không khả dụng)
        color: BGR color tuple
        thickness: Text thickness
    """
    # Nếu chỉ là ASCII, dùng cv2.putText thường
    try:
        text.encode('ascii')
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return
    except UnicodeEncodeError:
        # Text có ký tự đặc biệt (Việt, emoji, v.v.)
        pass
    
    # Dùng PIL để render Vietnamese text
    if VIETNAMESE_FONT is None:
        # Fallback: Dùng cv2 với encoded text
        logger.debug(f"Using fallback rendering for: {text}")
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return
    
    try:
        # Convert frame to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Draw text with Vietnamese font (color in RGB)
        rgb_color = (color[2], color[1], color[0])  # BGR to RGB
        draw.text(pos, text, font=VIETNAMESE_FONT, fill=rgb_color)
        
        # Convert back to OpenCV
        frame_rgb = np.array(pil_image)
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR, dst=frame)
    except Exception as e:
        logger.debug(f"PIL rendering failed, fallback to cv2: {e}")
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_mediapipe_skeleton(frame, keypoints, confidence_threshold=0.4):
    """Vẽ bộ xương dựa trên danh sách MEDIAPIPE_EDGES"""
    if keypoints is None or len(keypoints) == 0: 
        return

    for edge in MEDIAPIPE_EDGES:
        start_idx, end_idx = edge
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            kp1, kp2 = keypoints[start_idx], keypoints[end_idx]
            if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
                cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (0, 255, 0), 1)

    for kp in keypoints:
        if kp[2] > confidence_threshold:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)

def is_meaningful(val):
        if not val:
            return False
        val_str = str(val).lower().strip()
        return val_str not in ['unknown', 'n/a', 'chưa xác định', 'chua xac dinh', 'none', '']


def draw_info_panel(frame, bbox, attrs):
    """
    🔥 FIXED: Giữ nguyên logic cũ (valid_count, Analyzing..., text list, panel nền, skin tone)
    Nhưng vẽ ô màu áo/quần NGAY BÊN DƯỚI dòng Top/Bot (giống skin tone)
    Update realtime (không chờ consolidation lock)
    """
    if not attrs:
        return

    # print(f"🔥 draw_info_panel called with attrs: {attrs}")

    # ============================================================
    # BƯỚC 1: KIỂM TRA DỮ LIỆU CÓ Ý NGHĨA
    # ============================================================
    valid_attrs = {
        'gender': is_meaningful(attrs.get('gender')),
        'age': is_meaningful(attrs.get('age_onnx')),
        'country': is_meaningful(attrs.get('country')),
        'race': is_meaningful(attrs.get('race')),
        'emotion': is_meaningful(attrs.get('emotion')),
        'upper': is_meaningful(attrs.get('upper_type')),
        'lower': is_meaningful(attrs.get('lower_type'))
    }

    valid_count = sum(valid_attrs.values())
    
    # ============================================================
    # GỬI GENDER ĐẾN SERVER ĐỂ PHÁT VIDEO (VỚI COOLDOWN)
    # ============================================================
    # global last_video_request_time
    
    # if valid_attrs['gender']:
    #     gender = attrs.get('gender').lower()
    #     print(f"🔥 Detected gender: {gender}")
    #     current_time = time.time()
        
    #     # Chỉ gửi nếu đã hết thời gian cooldown
    #     if current_time - last_video_request_time >= VIDEO_COOLDOWN_SECONDS:
    #         # if 'male' in gender or 'Male' in gender:
    #         if gender == 'male':
    #             logger.info(f"👨 [VIDEO] Phát hiện nam giới, gửi đến server (Async)")
    #             # 🔥 ASYNC DISPATCH: Chạy trong thread để không block Main Loop
    #             # import threading
    #             # threading.Thread(
    #             #     target=send_gender_to_server,
    #             #     args=('male',),
    #             #     daemon=True
    #             # ).start()
    #             if gender == 'male':
    #                 logger.info("👨 Gửi male đồng bộ để debug")
    #                 success = send_gender_to_server('male')
    #                 print(f"DEBUG - Gửi gender thành công? {success}")
    #             # Optimistic cooldown update
    #             last_video_request_time = current_time
    #             logger.info(f"⏱️ [VIDEO] Cooldown {VIDEO_COOLDOWN_SECONDS}s bắt đầu")
    #         else:
    #             logger.info(f"👩 [VIDEO] Phát hiện nữ giới, gửi đến server (Async)")
    #             # 🔥 ASYNC DISPATCH
    #             import threading
    #             threading.Thread(
    #                 target=send_gender_to_server,
    #                 args=('female',),
    #                 daemon=True
    #             ).start()
                
    #             last_video_request_time = current_time
    #             logger.info(f"⏱️ [VIDEO] Cooldown {VIDEO_COOLDOWN_SECONDS}s bắt đầu")
    #     else:
    #         time_remaining = VIDEO_COOLDOWN_SECONDS - (current_time - last_video_request_time)
    #         logger.debug(f"⏳ [VIDEO COOLDOWN] Còn {time_remaining:.1f}s trước khi có thể gửi request mới")

    # Không có dữ liệu → skip
    if valid_count == 0:
        return

    x1, y1, x2, y2 = map(int, bbox)

    # Chỉ có 1-2 thuộc tính → hiển thị cảnh báo
    if valid_count <= 2:
        put_vietnamese_text(frame, f"Analyzing... ({valid_count}/7)",
                    (x2 + 10, y1 + 20),
                    0.5, (0, 255, 255), 1)
        return

    # ============================================================
    # BƯỚC 2: CHUẨN BỊ DANH SÁCH
    # ============================================================
    x1, y1, x2, y2 = map(int, bbox)
    info_x = x2 + 10
    current_y = y1 + 20
    line_h = 25

    display_list = []

    if valid_attrs['gender']:
        display_list.append(('gender', f"Gender: {attrs['gender']}", (255, 200, 0)))

    # Hiển thị country từ CCCD hoặc race từ AI detect
    if valid_attrs['country']:
        display_list.append(('country', f"Country: {attrs['country']}", (255, 0, 255)))
    elif valid_attrs['race']:
        display_list.append(('race', f"Race: {attrs['race']}", (255, 0, 255)))

    if valid_attrs['age']:
        display_list.append(('age', f"Age: {attrs['age_onnx']}", (0, 165, 255)))

    if valid_attrs['emotion']:
        display_list.append(('emotion', f"Emo: {attrs['emotion']}", (0, 0, 255)))

    if valid_attrs['upper']:
        display_list.append(('upper', f"Top: {attrs['upper_type']}", (0, 255, 0)))

    if valid_attrs['lower']:
        display_list.append(('lower', f"Bot: {attrs['lower_type']}", (0, 255, 255)))

    # ============================================================
    # BƯỚC 3: VẼ PANEL NỀN
    # ============================================================
    panel_h = len(display_list) * line_h + 10

    if attrs.get('skin_tone_bgr') is not None:
        panel_h += line_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (info_x, y1), (info_x + 220, y1 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (info_x, y1), (info_x + 220, y1 + panel_h), (255, 255, 255), 1)

    # ============================================================
    # BƯỚC 4: VẼ TEXT + Ô MÀU
    # ============================================================
    for attr_type, text, color in display_list:
        # Vẽ text label
        put_vietnamese_text(frame, text, (info_x + 8, current_y),
                    0.5, color, 1)

        # Ô màu ngay bên dưới text Top/Bot
        if attr_type == 'upper' and attrs.get('upper_color'):
            u_color = tuple(map(int, attrs['upper_color']))
            cv2.rectangle(frame, (info_x + 150, current_y + 4),
                          (info_x + 180, current_y + 18), u_color, -1)
            cv2.rectangle(frame, (info_x + 150, current_y + 4),
                          (info_x + 180, current_y + 18), (255, 255, 255), 1)

        if attr_type == 'lower' and attrs.get('lower_color'):
            l_color = tuple(map(int, attrs['lower_color']))
            cv2.rectangle(frame, (info_x + 150, current_y + 4),
                          (info_x + 180, current_y + 18), l_color, -1)
            cv2.rectangle(frame, (info_x + 150, current_y + 4),
                          (info_x + 180, current_y + 18), (255, 255, 255), 1)

        current_y += line_h

    # ============================================================
    # BƯỚC 5: VẼ SKIN TONE
    # ============================================================
    skin_bgr = attrs.get('skin_tone_bgr')
    if skin_bgr is not None:
        put_vietnamese_text(frame, "Skin:", (info_x + 8, current_y),
                    0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (info_x + 150, current_y + 4),
                      (info_x + 180, current_y + 18),
                      tuple(map(int, skin_bgr)), -1)
        cv2.rectangle(frame, (info_x + 150, current_y + 4),
                      (info_x + 180, current_y + 18), (255, 255, 255), 1)


def draw_tracked_objects(frame, tracked_objects, db_manager=None):
    """
    Vẽ tracked objects với prioritize CCCD metadata
    
    Args:
        frame: Frame hình ảnh
        tracked_objects: Dict tracked objects từ track_manager
        db_manager: VectorDatabase_Manager (optional) để lấy CCCD metadata
    """
    global last_door_open_time  # Sử dụng biến toàn cục
    
    for track_id, obj_data in tracked_objects.items():
        bbox = obj_data.get('bbox')
        if not bbox:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        status = obj_data.get('status', 'pending')
        final_id = obj_data.get('final_id', f"Temp_{track_id}")
        source = obj_data.get('identification_source', '')

        # Màu theo status
        color_map = {
            'pending': (255, 255, 0),
            'tentative': (0, 165, 255),
            'identified': (0, 140, 255),
            'confirmed': (0, 255, 0)
        }
        color = color_map.get(status, (255, 255, 255))

        # Lấy attributes - ưu tiên CCCD metadata
        attrs = {}
        
        # Kiểm tra DB metadata cho CCCD info - SỬ DỤNG CACHE
        if db_manager and final_id:
            try:
                # Use cache instead of direct DB query
                db_meta = metadata_cache.get(final_id, db_manager)
                if db_meta and db_meta.get('cccd_matched'):
                    # Ưu tiên CCCD metadata - hiển thị name, age, gender, country
                    attrs = {
                        'name': db_meta.get('cccd_name', 'Unknown'),
                        'gender': db_meta.get('cccd_gender', 'unknown'),
                        'age_onnx': db_meta.get('cccd_age', 'unknown'),
                        'country': db_meta.get('cccd_country', 'unknown'),
                        'emotion': 'N/A'
                    }
                    logger.debug(f"[DRAW] {final_id}: Using CCCD - Gender: {attrs['gender']}, Age: {attrs['age_onnx']}, Country: {attrs['country']}")
                    
#                     # ============================================================
#                     # CƠ CHẾ MỞ CỬA VỚI COOLDOWN
#                     # ============================================================
#                     current_time = time.time()
#                     
#                     # Kiểm tra 2 điều kiện:
#                     # 1. Chưa mở cửa cho người này (theo metadata)
#                     # 2. Đã hết thời gian cooldown kể từ lần mở cửa cuối cùng
#                     if (not db_meta.get('door_opened', False) and
#                         current_time - last_door_open_time >= DOOR_COOLDOWN_SECONDS):
#                         
#                         person_name = db_meta.get('cccd_name', final_id)
#                         logger.info(f"🔓 [DOOR] Mở cửa lần đầu cho {person_name} ({final_id}) - cooldown OK")
#                         
#                         # Gọi hàm mở cửa
#                         if open_door():
#                             # Cập nhật timestamp toàn cục
#                             last_door_open_time = current_time
#                             
#                             # Cập nhật metadata cá nhân
#                             updated_meta = db_meta.copy()
#                             updated_meta['door_opened'] = True
#                             updated_meta['door_opened_timestamp'] = current_time
#                             
#                             db_manager.update_metadata(final_id, updated_meta)
#                             logger.info(f"💾 [DOOR] Đã cập nhật metadata door_opened=True cho {final_id}")
#                     else:
#                         if db_meta.get('door_opened', False):
#                             logger.debug(f"⏭️ [DOOR] Đã mở cửa trước đó cho {final_id}")
#                         else:
#                             time_remaining = DOOR_COOLDOWN_SECONDS - (current_time - last_door_open_time)
#                             logger.debug(f"⏳ [DOOR COOLDOWN] Chưa hết {time_remaining:.1f}s kể từ lần mở cuối")
                    
            except Exception as e:
                logger.debug(f"[DRAW] Error reading DB metadata: {e}")
        
        # Fallback: Dùng final_attributes từ live analysis
        if not attrs or attrs.get('gender') == 'unknown':
            attrs = (obj_data.get('final_attributes') or {}).copy()

        # Lấy name từ attrs (ưu tiên), fallback từ cccd_name
        name = attrs.get('name') or obj_data.get('cccd_name', '')

        # Vẽ bbox
        thickness = 3 if status == 'confirmed' else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Vẽ label
        clean_source = ''
        if source:  # Check if source is not None
            clean_source = source.replace("HỢP NHẤT", "HOP NHAT") \
                                 .replace("TOÀN THÂN", "BODY") \
                                 .replace("MẶT", "FACE")

        if name and name not in ['Unknown', 'unknown', '']:
            # Khi có tên trong DB, chỉ hiển thị tên (lớn), không hiện ID
            put_vietnamese_text(frame, name, (x1, y1 - 30),
                        1.0, (0, 255, 255), 2)
        else:
            label = f"{final_id}"
            if clean_source:
                label += f" [{clean_source}]"
            elif status:
                label += f" [{status.upper()}]"
            put_vietnamese_text(frame, label, (x1, y1 - 10),
                        0.7, color, 2)

        # Vẽ skeleton
        skeleton = obj_data.get('last_keypoints') or obj_data.get('last_skeleton')
        if skeleton is not None:
            draw_mediapipe_skeleton(frame, skeleton)
        
        # Merge realtime từ history
        history = obj_data.get('history_attributes', [])
        if history:
            latest = history[-1]
            clothing = latest.get('clothing_analysis') or {}
            emotion = latest.get('emotion_analysis') or {}

            if 'upper_type' in clothing and is_meaningful(clothing['upper_type']):
                attrs['upper_type'] = clothing['upper_type']
            if 'lower_type' in clothing and is_meaningful(clothing['lower_type']):
                attrs['lower_type'] = clothing['lower_type']
            if 'upper_color' in clothing and clothing['upper_color']:
                attrs['upper_color'] = clothing['upper_color']
            if 'lower_color' in clothing and clothing['lower_color']:
                attrs['lower_color'] = clothing['lower_color']
            if 'skin_tone_bgr' in clothing and clothing['skin_tone_bgr']:
                attrs['skin_tone_bgr'] = clothing['skin_tone_bgr']
            if 'emotion' in emotion and is_meaningful(emotion['emotion']):
                attrs['emotion'] = emotion['emotion']
        
        # Vẽ panel
        draw_info_panel(frame, bbox, attrs)

        # Body quality
        last_result = obj_data.get('history_attributes', [])
        if last_result and len(last_result) > 0:
            latest = last_result[-1]
            body_quality = latest.get('body_quality_score', 0.0)

            if body_quality > 0:
                if body_quality < 0.4:
                    quality_color = (0, 0, 255)
                elif body_quality < 0.6:
                    quality_color = (0, 165, 255)
                else:
                    quality_color = (0, 255, 0)

                put_vietnamese_text(
                    frame,
                    f"Body Q: {body_quality:.2f}",
                    (x1, y2 + 40),
                    0.5,
                    quality_color,
                    1
                )
    return frame

