# core/tracker/manager.py - COMPLETE FILE
import os
import queue
import threading
import time
from collections import deque

import cv2
import config
from .utils import TrackerUtils
from .consolidation import AttributeConsolidator
from .matching import ConfirmedPersonMatcher
import logging
import numpy as np
from utils.focus_quality_checker import FocusQualityChecker

logger = logging.getLogger(__name__)

DEBUG_FOLDER = "debug_aligned"
SAVE_DEBUG_IMAGES = True

if SAVE_DEBUG_IMAGES and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

class TrackManager:
    """
    🔥 FIXED VERSION với Focus Quality Gating
    - Chỉ chốt ID khi camera lấy nét tốt
    - Xóa duplicate code
    - Lưu body_crop đầy đủ
    """

    def __init__(self, analyzer, db_manager, dual_stream_manager=None, llm_sender=None):
        self.analyzer = analyzer
        self.db_manager = db_manager
        self.consolidator = AttributeConsolidator(db_manager)
        self.matcher = ConfirmedPersonMatcher(db_manager)
        
        # LLM Integration
        self.dual_stream_manager = dual_stream_manager
        self.llm_sender = llm_sender
        
        # 🔥 Focus Quality Checker
        self.focus_checker = FocusQualityChecker(
            min_face_sharpness=120,
            min_body_sharpness=80,
            min_face_size=80,
            min_body_area=30000
        )
        
        self.cancelled_ids = set()
        self.tracked_objects = {}
        self.next_person_id = 1
        self.id_lock = threading.Lock()
        
        self.global_frame_count = 0
        self.last_queue_clear_frame = 0
    
    def _identify_or_register(self, track_id):
        """
        🔥 IMPROVED: Them rematch check truoc khi search DB
        - Kiem tra confirmed persons da thay trong 5 giay gan do
        - Chi search DB neu rematch that bai
        """
        if track_id not in self.tracked_objects: 
            return

        obj_data = self.tracked_objects[track_id]
        current_final_id = obj_data.get('final_id', '')

        # ============================================================
        # BƯỚC 0: DEADLOCK CHECK
        # ============================================================
        if obj_data['status'] == 'confirmed' and not current_final_id.startswith('Temp_'):
            return
        
        history = obj_data.get('history_attributes', [])
        if len(history) == 0:
            return
        
        # ============================================================
        # BƯỚC 0.5: CHECK CONFIRMED PERSON REMATCH (NEW)
        # ============================================================
        bbox = obj_data.get('bbox', [0, 0, 100, 100])
        reid_query = TrackerUtils.get_query_vector(obj_data['reid_vectors'])
        face_query = TrackerUtils.get_query_vector_face(obj_data['face_vectors'])
        final_attrs = obj_data.get('final_attributes', {})
        
        rematch_result = self.matcher.check_confirmed_person_rematch(
            track_id=track_id,
            bbox=bbox,
            face_vector=face_query,
            reid_vector=reid_query,
            current_attributes=final_attrs
        )
        
        if rematch_result:
            # Tim thay confirmed person cu
            final_id, rematch_score, rematch_source = rematch_result
            obj_data.update({
                'final_id': final_id,
                'identification_source': rematch_source,
                'status': 'identified'
            })
            
            logger.info(
                f"✅ [REMATCH CONFIRMED] Track {track_id} → {final_id} "
                f"({rematch_source}) Score: {rematch_score:.4f}"
            )
            
            # Luu vector ReID
            if list(obj_data['reid_vectors']):
                self.db_manager.add_vectors(config.REID_NAMESPACE, final_id, list(obj_data['reid_vectors']))
                self.db_manager.save_all_databases()
            
            self.consolidator.consolidate(obj_data)
            return  # Dung lai, khong can search DB
        
        # ============================================================
        # BƯỚC 0.6: CHECK LLM PROCESSING (NEW)
        # ============================================================
        # Neu rematch thanh cong, kiem tra xem co can gui LLM khong
        if rematch_result:
            final_id = rematch_result[0]
            need_llm, duration, reason = self.matcher.check_llm_processing_needed(final_id)
            
            if need_llm and self.llm_sender and self.dual_stream_manager:
                # Lay metadata day du
                metadata = self.db_manager.get_metadata(final_id)
                
                # Chuẩn bị data cho LLM
                llm_data = {
                    'person_id': final_id,
                    'duration_minutes': duration,
                    'first_seen': metadata.get('first_seen_time'),
                    'last_seen': metadata.get('last_seen_time'),
                    'attributes': {
                        'gender': metadata.get('confirmed_gender', 'unknown'),
                        'age': metadata.get('confirmed_age', 'unknown'),
                        'race': metadata.get('confirmed_race', 'unknown')
                    },
                    'context': 'long_absence',
                    'reason': reason
                }
                
                logger.warning(
                    f"🤖 [LLM SEND] {final_id} - Duration: {duration:.1f} min\n"
                    f"   Reason: {reason}\n"
                    f"   Attributes: {llm_data['attributes']}\n"
                    f"   → Sending to LLM for processing"
                )
                
                # Gửi qua dual_stream_manager
                try:
                    # Goi send_long_absence thong qua dual_stream_manager
                    if hasattr(self.dual_stream_manager, 'send_long_absence_notification'):
                        self.dual_stream_manager.send_long_absence_notification(
                            person_id=final_id,
                            duration_minutes=duration,
                            metadata=llm_data,
                            llm_sender=self.llm_sender
                        )
                    else:
                        # Fallback: goi truc tiep llm_sender
                        if hasattr(self.llm_sender, 'send_custom_message'):
                            message = (
                                f"Person {final_id} returned after {duration:.1f} minutes.\n"
                                f"Gender: {llm_data['attributes']['gender']}, "
                                f"Age: {llm_data['attributes']['age']}, "
                                f"Race: {llm_data['attributes']['race']}"
                            )
                            self.llm_sender.send_custom_message(message)
                    
                    logger.info(f"✅ [LLM SENT] Successfully sent {final_id} to LLM")
                    
                except Exception as e:
                    logger.error(f"❌ [LLM ERROR] Failed to send {final_id}: {e}")
        
        # ============================================================
        # BƯỚC 1: FALLBACK SEARCH DB (neu rematch that bai)
        # ============================================================
        # LOGIC: Neu rematch fail nhung co face vector, vay search DB
        # voi logic nhu sau:
        # - Neu search DB tim thay → co the la confirmed person khac
        #   hoac nguoi toan toan moi
        # - Neu search DB khong tim thay → tao ID moi
        
        if face_query is None and reid_query is None: 
            return

        # Log rematch failure reason
        logger.info(
            f"⏳ [REMATCH SKIP] Track {track_id} - Spatial/Temporal not match, "
            f"fallback to DB search"
        )

        face_match = self.db_manager.search_vector_with_voting(
            config.FACE_NAMESPACE, face_query
        ) if face_query else None
        
        reid_match = self.db_manager.search_vector_with_voting(
            config.REID_NAMESPACE, reid_query
        ) if reid_query else None

        logger.info(f"🔍 [DB SEARCH RESULT] Track {track_id}: Face={face_match} | ReID={reid_match}")
        
        f_id, f_score = face_match if face_match else (None, 0.0)
        r_id, r_score = reid_match if reid_match else (None, 0.0)

        # ============================================================
        # BƯỚC 2: ADAPTIVE THRESHOLD
        # ============================================================
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        dynamic_threshold = 0.50
        if bbox_area > 80000: dynamic_threshold = 0.60
        elif len(history) > 15: dynamic_threshold = 0.48
        elif bbox_area < 30000: dynamic_threshold = 0.45

        # ============================================================
        # BƯỚC 3: QUYẾT ĐỊNH ID - 🔥 FACE PRIORITY (Face unique hơn quần áo)
        # ============================================================
        final_id, final_score, final_source = None, 0.0, ""

        # 🔥 BƯỚC 3A: FACE HỢP NHẤT (Cả Face và ReID cùng agree)
        if f_id == r_id and f_id is not None:
            final_id, final_score, final_source = f_id, max(f_score, r_score), "HỢP NHẤT (FACE+REID)"

        # 🔥 BƯỚC 3B: FACE PRIORITY - Nếu Face match tin cậy → dùng Face
        # (Vì Face không thay đổi khi pose/angle/đồ thay đổi)
        elif f_id and f_score > 0.65:
            final_id, final_score, final_source = f_id, f_score, "MẶT (PRIORITY)"
            
            # ⚠️ Log nếu ReID suggest người khác (có thể pose/angle khác)
            if r_id and r_id != f_id and r_score > 0.75:
                logger.warning(
                    f"⚠️ [FACE OVERRIDE] Track {track_id}: "
                    f"Face={f_id} (score={f_score:.4f}), "
                    f"but ReID={r_id} (score={r_score:.4f}). "
                    f"Using FACE (more reliable than ReID)"
                )

        # 🔥 BƯỚC 3C: FALLBACK ReID (Khi không có Face match tin cậy)
        elif r_score > 0.80:
            final_id, final_score, final_source = r_id, r_score, "TOÀN THÂN (VERY HIGH)"
        elif r_score > 0.60:
            final_id, final_score, final_source = r_id, r_score, "TOÀN THÂN (HIGH)"
        elif r_score > dynamic_threshold:
            final_id, final_score, final_source = r_id, r_score, "TOÀN THÂN (MODERATE)"
        elif r_id:
            final_id, final_score, final_source = r_id, r_score, "TOÀN THÂN (LOW)"

        # ============================================================
        # BƯỚC 4: CẬP NHẬT STATUS
        # ============================================================
        if final_id:
            # ✅ TÌM THẤY NGƯỜI CŨ
            obj_data.update({
                'final_id': final_id,
                'identification_source': final_source,
                'status': 'identified'
            })
            
            logger.info(f"✅ [MATCHED] Track {track_id} → {final_id} ({final_source}) Score: {final_score:.4f}")
            
            # Luu vector ReID
            if list(obj_data['reid_vectors']):
                self.db_manager.add_vectors(config.REID_NAMESPACE, final_id, list(obj_data['reid_vectors']))
                self.db_manager.save_all_databases()
            
            # Goi consolidator
            logger.info(f"🔄 [CONSOLIDATE] Calling for matched person: {final_id}")
            self.consolidator.consolidate(obj_data)
        
        else:
            # ✅ TẠO PROFILE MỚI (Person_X)
            valid_frames = len(history)
            
            if bbox_area > config.MIN_BBOX_AREA and valid_frames >= 2: 
                with self.id_lock:
                    new_id = f"Person_{self.next_person_id}"
                    self.next_person_id += 1
                
                # 🔥 LUÔN SET 'identified', ĐỂ CONSOLIDATOR QUYẾT ĐỊNH 'confirmed'
                obj_data.update({
                    'final_id': new_id, 
                    'status': 'identified'  # Consolidator sẽ chuyển lên 'confirmed' khi đủ attributes
                })
                
                logger.info(
                    f"🆕 [NEW PROFILE] Track {track_id} → {new_id} "
                    f"(Frames: {valid_frames}, Area: {bbox_area}) | Status: identified"
                )                
                
                # Lưu vector ReID
                if list(obj_data['reid_vectors']):
                    self.db_manager.add_vectors(config.REID_NAMESPACE, new_id, list(obj_data['reid_vectors']))
                    self.db_manager.save_all_databases()
                
                # 🔥 GỌI CONSOLIDATOR CHO NGƯỜI MỚI
                logger.info(f"🔄 [CONSOLIDATE] Calling for new person: {new_id}")
                self.consolidator.consolidate(obj_data)
            
            else:
                logger.debug(f"⏳ [WAITING] Track {track_id} needs more data")



    def process_attribute_results(self, attribute_result_queue):
        """
        🔥 TRIGGER ID NGAY KHI CÓ ĐỦ DỮ LIỆU
        """
        if attribute_result_queue.empty():
            return
        
        while not attribute_result_queue.empty():
            track_id, analysis_result = attribute_result_queue.get()
            
            if track_id not in self.tracked_objects or not analysis_result: 
                continue
            
            if analysis_result.get('status') == 'skipped_blur':
                continue

            obj_data = self.tracked_objects[track_id]
            person_id = obj_data.get('final_id', f"Track_{track_id}")

            # ============================================================
            # BƯỚC 1: APPEND HISTORY
            # ============================================================
            obj_data['history_attributes'].append(analysis_result)
            
            print(f"✅ [APPEND] Track {track_id} | History len: {len(obj_data['history_attributes'])}")
            
            # 🔥 CONSOLIDATE NGAY SAU KHI APPEND
            print(f"🔄 [CALLING CONSOLIDATE] For {person_id}")
            self.consolidator.consolidate(obj_data)
            
            # ============================================================
            # BƯỚC 2: LƯU VECTORS
            # ============================================================
            face_vec = analysis_result.get('face_vector')
            reid_vec = analysis_result.get('reid_vector')
            
            if obj_data['status'] in ['pending', 'tentative']:
                if reid_vec and isinstance(reid_vec, np.ndarray):
                    obj_data['reid_vectors'].append(reid_vec)
                    print(f"   → ReID vectors: {len(obj_data['reid_vectors'])}")
                
                if face_vec:
                    face_conf = analysis_result.get('face_conf', 0)
                    if face_conf >= 0.5:
                        obj_data['face_vectors'].append((face_vec, face_conf))
                        print(f"   → Face vectors: {len(obj_data['face_vectors'])}")

            # ============================================================
            # 🔥 BƯỚC 3: TRIGGER ID - CHO PHÉP RE-IDENTIFY (WITH THROTTLING)
            # ============================================================
            # 🔥 FIX: Cho phép gọi identify cả khi status='identified'
            # để có cơ hội chuyển lên 'confirmed'
            # 🔥 THROTTLE: Chỉ gọi mỗi 30 frames để tránh spam DB queries
            if obj_data['status'] in ['pending', 'identified']:
                history_len = len(obj_data['history_attributes'])
                has_vectors = len(obj_data['reid_vectors']) > 0 or len(obj_data['face_vectors']) > 0
                
                # Initialize last_identify_frame if not exists
                if 'last_identify_frame' not in obj_data:
                    obj_data['last_identify_frame'] = 0
                    logger.info(f"🆕 [THROTTLE INIT] {person_id} - Initialized last_identify_frame=0")
                
                frames_since_last_id = self.global_frame_count - obj_data['last_identify_frame']
                should_try_identify = False
                
                logger.info(
                    f"🔍 [CHECK TRIGGER] {person_id} | "
                    f"History: {history_len} | Vectors: {has_vectors} | "
                    f"Status: {obj_data['status']} | "
                    f"Global Frame: {self.global_frame_count} | "
                    f"Last ID Frame: {obj_data['last_identify_frame']} | "
                    f"Frames since ID: {frames_since_last_id}"
                )
                
                # 🔥 ĐIỀU KIỆN 1: Có ≥ 1 frames + có vectors
                if history_len >= 1 and has_vectors:
                    # First time or throttled retry
                    if obj_data['last_identify_frame'] == 0:
                        should_try_identify = True
                        logger.warning(f"🎯 [FIRST IDENTIFY] Track {track_id} - First time!")
                    elif frames_since_last_id >= 30:
                        should_try_identify = True
                        logger.warning(f"🎯 [RETRY IDENTIFY] Track {track_id} - Retry after {frames_since_last_id} frames")
                    else:
                        logger.warning(f"⏭️ [THROTTLE SKIP] {person_id} - Waiting {30 - frames_since_last_id} more frames")
                
                # 🔥 ĐIỀU KIỆN 2: Có ≥ 3 frames (fallback)
                elif history_len >= 2:
                    if obj_data['last_identify_frame'] == 0:
                        should_try_identify = True
                        logger.warning(f"🎯 [FIRST IDENTIFY FALLBACK] Track {track_id}")
                    elif frames_since_last_id >= 30:
                        should_try_identify = True
                        logger.warning(f"🎯 [RETRY IDENTIFY FALLBACK] Track {track_id} - After {frames_since_last_id} frames")
                    else:
                        logger.warning(f"⏭️ [THROTTLE SKIP FALLBACK] {person_id} - Waiting {30 - frames_since_last_id} more frames")
                else:
                    logger.info(f"   ⏳ [WAITING] {person_id} - Need more data (history={history_len}, vectors={has_vectors})")
                
                # Execute identification if allowed
                if should_try_identify:
                    logger.warning(f"🚀 [EXECUTING IDENTIFY] {person_id} at frame {self.global_frame_count}")
                    obj_data['last_identify_frame'] = self.global_frame_count
                    self._identify_or_register(track_id)
                    logger.warning(f"✅ [IDENTIFY DONE] {person_id} - Updated last_identify_frame to {self.global_frame_count}")

            # ============================================================
            # BƯỚC 4: LƯU VECTORS VÀO DB (CHỈ IDENTIFIED+)
            # ============================================================
            if not person_id.startswith("Temp_"):
                # Save face vectors to DB
                if face_vec is not None and obj_data.get('status') in ['identified', 'confirmed']:
                    face_count = self.db_manager.count_vectors_for_id(config.FACE_NAMESPACE, person_id)
                    if face_count < config.MAX_FACE_VECTORS_PER_PROFILE:
                        self.db_manager.add_vectors(config.FACE_NAMESPACE, person_id, [face_vec])
                        print(f"   💾 [DB] Saved face vector for {person_id} ({face_count+1}/{config.MAX_FACE_VECTORS_PER_PROFILE})")

            # ============================================================
            # BƯỚC 5: CONSOLIDATE
            # ============================================================
            self.consolidator.consolidate(obj_data)
            
            # ============================================================
            # BƯỚC 6: LƯU CONFIRMED ATTRIBUTES (immutable)
            # ============================================================
            # Neu status vua chuyen thanh 'confirmed', luu attributes bat bien
            if obj_data['status'] == 'confirmed' and not person_id.startswith("Temp_"):
                # Kiem tra xem attributes co thay doi khong
                # (Neu co thay doi thi khong luu - chi luu lan dau tien khi confirm)
                meta = self.db_manager.get_metadata(person_id)
                if 'confirmed_gender' not in meta:
                    # Lan dau tien confirmed -> luu bat bien
                    self.matcher.save_confirmed_attributes(person_id, obj_data)
                    # Lay lai metadata sau khi save de lay confirmed_name
                    meta = self.db_manager.get_metadata(person_id)
                    confirmed_name = meta.get('confirmed_name', 'Unknown')
                    logger.info(f"✅ [CONFIRMED] {person_id} - {confirmed_name} - Saved immutable attributes to DB")
                else:
                    # Da luu roi, chi update last_seen (khong can bbox)
                    meta['last_seen_time'] = __import__('datetime').datetime.now().isoformat()
                    self.db_manager.update_metadata(person_id, meta)
                    confirmed_name = meta.get('confirmed_name', 'Unknown')
                    logger.debug(f"🔄 [UPDATE] {person_id} - {confirmed_name} - Updated last_seen")


    def update_tracks(self, track_ids, bboxes, frame, attribute_task_queue, frame_original=None, scale_x=1.0, scale_y=1.0):
        """
        🔥 FULLY FIXED với Frame Skip Controller integration + High Res Support
        """
        self.global_frame_count += 1
        current_track_ids = set(track_ids)
        current_time = time.time()
        
        # ============================================================
        # KHÔNG CÓ NGƯỜI → XÓA QUEUE
        # ============================================================
        if len(track_ids) == 0:
            if attribute_task_queue is not None:
                cleared_count = 0
                try:
                    while not attribute_task_queue.empty():
                        attribute_task_queue.get_nowait()
                        cleared_count += 1
                except:
                    pass
                
                if cleared_count > 0 and (self.global_frame_count - self.last_queue_clear_frame) > 30:
                    logger.info(f"🗑️ [QUEUE] Đã xóa {cleared_count} tasks vì không còn người")
                    self.last_queue_clear_frame = self.global_frame_count
            return
        
        # ============================================================
        # PROCESSING TRACKS
        # ============================================================
        for i, track_id in enumerate(track_ids):
            bbox = bboxes[i]
            
            if track_id not in self.tracked_objects:
                logger.info(f"✨ [ID: {track_id}] Track mới.")
                self.tracked_objects[track_id] = {
                    'status': 'pending', 
                    'final_id': f"Temp_{track_id}", 
                    'bbox': bbox,
                    'reid_vectors': deque(maxlen=config.MOVING_AVERAGE_WINDOW),
                    'face_vectors': deque(maxlen=config.MOVING_AVERAGE_WINDOW),
                    'disappeared_frames': 0, 
                    'quality_score': 0.0,
                    'history_attributes': deque(maxlen=30),
                    'final_attributes': None,
                    'frames_since_last_attr_analysis': 4,
                    'last_analysis_frame': 0
                }
            
            obj_data = self.tracked_objects[track_id]
            obj_data['bbox'] = bbox
            obj_data['disappeared_frames'] = 0
            obj_data['frames_since_last_attr_analysis'] += 1
            
            should_send_task = False
            
            # HYBRID MODE THROTTLE LOGIC
            current_status = obj_data['status']
            
            if current_status == 'pending':
                # PENDING: Skip every 5 frames @ 24 FPS = ~5 FPS effective (SMOOTH)
                if obj_data['frames_since_last_attr_analysis'] >= 2:
                    should_send_task = True
                    throttle_mode = "SMOOTH (every 5 frames, ~5 FPS)"
                else:
                    should_send_task = False
                    throttle_mode = f"SKIP (waiting {5 - obj_data['frames_since_last_attr_analysis']} more)"

                
            elif current_status == 'identified':
                # IDENTIFIED: Skip every 5 frames @ 24 FPS = ~5 FPS effective (SMOOTH)
                if obj_data['frames_since_last_attr_analysis'] >= 5:
                    should_send_task = True
                    throttle_mode = "SMOOTH (every 5 frames, ~5 FPS)"
                else:
                    should_send_task = False
                    throttle_mode = f"SKIP (waiting {5 - obj_data['frames_since_last_attr_analysis']} more)"
                    
            elif current_status == 'confirmed':
                # 🔥 CONFIRMED: Mỗi 5 frames (1s) - LIGHT update cho 5 FPS
                # Hoặc SKIP hoàn toàn nếu bạn chọn Option 1
                if obj_data['frames_since_last_attr_analysis'] >= 2:
                    should_send_task = True
                    throttle_mode = "LIGHT (every 5 frames, 5fps)"
                else:
                    throttle_mode = f"LIGHT (waiting {5 - obj_data['frames_since_last_attr_analysis']} more)"
            else:
                # Fallback cho tentative hoac status khac
                if obj_data['frames_since_last_attr_analysis'] >= 1:
                    should_send_task = True
                    throttle_mode = "FALLBACK (every frame)"
                else:
                    throttle_mode = "FALLBACK (waiting)"
            
            # Logging rõ ràng (mỗi 10 frames để thấy skip behavior)
            if self.global_frame_count % 10 == 0 or not should_send_task:
                person_id = obj_data.get('final_id', f"Track_{track_id}")
                skip_indicator = "⏭️ [SKIP]" if not should_send_task else "✅ [SEND]"
                logger.info(
                    f"{skip_indicator} {person_id} | Status: {current_status} | "
                    f"Mode: {throttle_mode} | Frames since last: {obj_data['frames_since_last_attr_analysis']}"
                )

            # ============================================================
            # 🔥 GỬI TASK VỚI FRAME SKIP LOGIC
            # ============================================================
            if should_send_task and attribute_task_queue is not None:
                current_queue_size = attribute_task_queue.qsize()
                
                # 🔥 TÍNH BBOX AREA
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1) 

                if current_queue_size > 15:
                    if self.global_frame_count % 10 == 0:
                        logger.warning(f"⏭️ [SKIP] Queue quá đầy ({current_queue_size}), skip frame")
                    continue
                   
                # 🔥 KIỂM TRA FRAME SKIP CONTROLLER
                if hasattr(self, 'frame_skip_controller'):
                    if not self.frame_skip_controller.should_process_frame(
                        track_id=track_id,
                        bbox_area=bbox_area,
                        queue_size=current_queue_size
                    ):
                        continue  # Skip frame này
                
                # CHECK FULLY LOCKED
                target_id = obj_data.get('final_id', f"Temp_{track_id}")
                
                if hasattr(self, 'dual_stream_manager'):
                    if self.dual_stream_manager.is_fully_locked(target_id):
                        if self.global_frame_count % 30 == 0:
                            logger.info(f"⏭️ [SKIP] {target_id} đã fully locked")
                        continue
                
                # 🔥 TẠO TASK DATA VỚI FULL RESOLUTION SUPPORT
                
                # 1. Tính toán bbox trên frame gốc 2K (nếu có)
                bbox_original = bbox  # Mặc định dùng bbox resized nếu không có frame gốc
                frame_for_face = frame.copy() # Mặc định dùng frame resized

                if frame_original is not None:
                    # Scale bbox từ 640x480 -> 2K
                    bx1 = int(x1 * scale_x)
                    by1 = int(y1 * scale_y)
                    bx2 = int(x2 * scale_x)
                    by2 = int(y2 * scale_y)
                    
                    # Clamp coordinates
                    h_orig, w_orig = frame_original.shape[:2]
                    bx1 = max(0, min(bx1, w_orig))
                    by1 = max(0, min(by1, h_orig))
                    bx2 = max(0, min(bx2, w_orig))
                    by2 = max(0, min(by2, h_orig))
                    
                    bbox_original = [bx1, by1, bx2, by2]
                    frame_for_face = frame_original # 🔥 Quan trọng: Pass reference (copy tốn RAM, worker sẽ copy nếu cần)
                
                task_data = {
                    'track_id': track_id,
                    'frame_resized': frame.copy(),          # 640x480: Dùng cho Pose, Clothing (nhẹ)
                    'frame_original': frame_for_face,       # 2K: Dùng cho Face (chất lượng cao)
                    'bbox_resized': bbox,                   # bbox 640x480
                    'bbox_original': bbox_original,         # bbox 2K
                    'person_id': target_id,
                    'confirmed_status': obj_data['status'],
                    'created_at_frame': self.global_frame_count,
                    'timestamp': current_time
                }
                
                try:
                    attribute_task_queue.put_nowait(task_data)
                    obj_data['frames_since_last_attr_analysis'] = 0
                    obj_data['last_analysis_frame'] = self.global_frame_count
                except queue.Full:
                    if self.global_frame_count % 30 == 0:
                        logger.warning(f"⚠️ [QUEUE] Không thể gửi task, queue đầy")

        # ============================================================
        # CLEANUP DISAPPEARED TRACKS
        # ============================================================
        disappeared_ids = set(self.tracked_objects.keys()) - current_track_ids
        for track_id in disappeared_ids:
            self.tracked_objects[track_id]['disappeared_frames'] += 1

        max_frames = 5 if len(current_track_ids) == 0 else config.MAX_DISAPPEARED_FRAMES

        cleanup_ids = [
            tid for tid, data in self.tracked_objects.items() 
            if data['disappeared_frames'] > max_frames
        ]

        for tid in cleanup_ids:
            person_id = self.tracked_objects[tid].get('final_id', f"Temp_{tid}")
            logger.info(
                f"🗑️ [CLEANUP] Đã xóa track {tid} ({person_id}) "
                f"sau {self.tracked_objects[tid]['disappeared_frames']} frames"
            )
            del self.tracked_objects[tid]
            
            if not person_id.startswith("Temp_"):
                self.db_manager.save_all_databases()