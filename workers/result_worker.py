"""
Result Processing Worker - Worker 3
Processes AI analysis results and updates track manager
"""

import time
import queue
import logging
import traceback
from collections import deque
import config

logger = logging.getLogger(__name__)


def result_processing_worker(result_queue, track_manager, reid_quality_stats, face_quality_stats):
    """
    🔥 WORKER 3 - FIXED VERSION
    
    Thứ tự xử lý ĐÚNG:
    1. Lưu vectors vào RAM
    2. Append vào history
    3. 🔥 CONSOLIDATE → Tạo final_attributes
    4. Kiểm tra điều kiện → Identify
    
    KHÔNG BAO GIỜ identify khi chưa có final_attributes!
    
    Args:
        result_queue: Queue containing analysis results
        track_manager: TrackManager instance
        reid_quality_stats: ReID quality statistics dict
        face_quality_stats: Face quality statistics dict
    """
    logger.info("🚀 [WORKER 3] Result Processing Worker started")
    
    while True:
        try:
            try:
                track_id, result = result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                # ============================================================
                # VALIDATE TRACK TỒN TẠI
                # ============================================================
                if track_id not in track_manager.tracked_objects:
                    logger.warning(f"⚠️ [WORKER 3] Track {track_id} not found, skipping")
                    result_queue.task_done()
                    continue
                
                obj_data = track_manager.tracked_objects[track_id]
                person_id = obj_data.get('final_id', f'Temp_{track_id}')
                
                logger.info(f"📤 [WORKER 3] Processing result for Track {track_id} ({person_id})")
                
                # ============================================================
                # EXTRACT DATA TỪ RESULT
                # ============================================================
                face_vec = result.get('face_vector')
                face_conf = result.get('face_conf', 0)
                reid_vec = result.get('reid_vector')
                
                # ============================================================
                # 📊 TRACK QUALITY STATISTICS
                # ============================================================
                # ReID Quality Tracking
                reid_quality_stats['total'] += 1
                if reid_vec is not None:
                    reid_quality_stats['accepted'] += 1
                else:
                    reason = result.get('body_quality_reason', '')
                    if 'blur' in reason.lower():
                        reid_quality_stats['rejected_blur'] += 1
                    elif 'dark' in reason.lower():
                        reid_quality_stats['rejected_dark'] += 1
                    elif 'contrast' in reason.lower():
                        reid_quality_stats['rejected_contrast'] += 1
                    elif 'small' in reason.lower():
                        reid_quality_stats['rejected_small'] += 1
                    elif 'overexposed' in reason.lower():
                        reid_quality_stats['rejected_overexposed'] += 1
                
                # Face Quality Tracking
                face_quality_stats['total'] += 1
                if face_vec is not None:
                    face_quality_stats['accepted'] += 1
                else:
                    if result.get('status') == 'skipped_blur':
                        face_quality_stats['rejected_blur'] += 1
                    elif face_conf < 0.65:
                        face_quality_stats['rejected_low_conf'] += 1
                
                # ============================================================
                # BƯỚC 1: LƯU VECTORS VÀO RAM
                # ============================================================
                MAX_VECTORS = 10
                
                # Lưu ReID vector
                if reid_vec is not None:
                    if len(obj_data['reid_vectors']) < MAX_VECTORS:
                        obj_data['reid_vectors'].append(reid_vec)
                        logger.debug(f"   → ReID vectors: {len(obj_data['reid_vectors'])}/{MAX_VECTORS}")
                
                # Lưu Face vector
                if face_vec is not None and face_conf >= 0.5:
                    if len(obj_data['face_vectors']) < MAX_VECTORS:
                        obj_data['face_vectors'].append((face_vec, face_conf))
                        logger.debug(f"   → Face vectors: {len(obj_data['face_vectors'])}/{MAX_VECTORS}")
                
                # ============================================================
                # BƯỚC 2: LƯU VÀO HISTORY
                # ============================================================
                MAX_HISTORY = 30

                try:
                    # 🔥 APPEND TRƯỚC - TRIM SAU
                    obj_data['history_attributes'].append(result)
                    
                    # 🔥 Tự động giữ đúng MAX_HISTORY frames
                    while len(obj_data['history_attributes']) > MAX_HISTORY:
                        obj_data['history_attributes'].popleft()
                    
                    history_len = len(obj_data['history_attributes'])
                    
                    # 🔥 LOG CHI TIẾT HƠN
                    logger.info(
                        f"   → History: {history_len}/{MAX_HISTORY} frames | "
                        f"ReID: {len(obj_data['reid_vectors'])} | "
                        f"Face: {len(obj_data['face_vectors'])}"
                    )
                    
                except Exception as e:
                    logger.error(f"❌ [HISTORY ERROR] Track {track_id}: {e}")
                    # 🔥 FALLBACK: Khởi tạo lại nếu bị lỗi
                    obj_data['history_attributes'] = deque(maxlen=MAX_HISTORY)
                    obj_data['history_attributes'].append(result)

                # ============================================================
                # 🔥 BƯỚC 3: CONSOLIDATE NGAY - QUAN TRỌNG NHẤT!
                # ============================================================
                
                # 🔥 SKIP CONSOLIDATION CHO CONFIRMED (đã stable rồi)
                if obj_data['status'] == 'confirmed':
                    logger.debug(f"⏭️ [SKIP CONSOLIDATE] {person_id} already confirmed - skip consolidation")
                else:
                    logger.info(f"🔄 [CONSOLIDATE] Calling for {person_id} (Status: {obj_data['status']})")
                    
                    # GỌI CONSOLIDATE ĐỂ TẠO final_attributes
                    track_manager.consolidator.consolidate(obj_data)
                    if obj_data.get('final_attributes'):
                        attrs = obj_data['final_attributes']
                        
                        # Thử gửi SEND-1 (chỉ gửi 1 lần)
                        track_manager.dual_stream_manager.send_stage1_ai_attributes(
                            person_id=person_id,
                            ai_attributes=attrs,
                            llm_sender=track_manager.llm_sender,
                            obj_data=obj_data
                        )
                    # Check xem đã có final_attributes chưa
                        logger.info(f"   ✅ final_attributes created:")
                        logger.info(f"      Gender: {obj_data['final_attributes'].get('gender')}")
                        logger.info(f"      Age: {obj_data['final_attributes'].get('age_onnx')}")
                        logger.info(f"      Race: {obj_data['final_attributes'].get('race')}")
                    else:
                        logger.warning(f"   ⚠️ final_attributes still None after consolidate!")
                
                # ============================================================
                # 🔥 BƯỚC 4: IDENTIFY CHỈ KHI ĐỦ ĐIỀU KIỆN
                # ============================================================
                if obj_data['status'] == 'pending':
                    history_len = len(obj_data['history_attributes'])
                    has_any_vector = (
                        len(obj_data['reid_vectors']) > 0 or 
                        len(obj_data['face_vectors']) > 0
                    )
                    
                    # 🔥 CHỈ CẦN 1 FRAME + CÓ VECTOR → IDENTIFY NGAY
                    if history_len >= 1 and has_any_vector:
                        logger.info(f"⚡ [INSTANT IDENTIFY] Track {track_id} after 1st result")
                        track_manager._identify_or_register(track_id)

                    # ĐIỀU KIỆN IDENTIFY: Có ít nhất 2 frames + có vectors
                    has_vectors = (
                        len(obj_data['reid_vectors']) > 0 or 
                        len(obj_data['face_vectors']) > 0
                    )

                    if history_len >= 2 and has_vectors:
                        logger.info(f"🎯 [TRIGGER IDENTIFY] Track {track_id} - Ready!")
                        track_manager._identify_or_register(track_id)
                        
                        # Log status sau khi identify
                        new_status = obj_data.get('status', 'unknown')
                        new_id = obj_data.get('final_id', 'unknown')
                        logger.info(f"   → New status: {new_status} | ID: {new_id}")
                    else:
                        logger.debug(
                            f"⏳ [WAITING] Track {track_id} needs more data\n"
                            f"   Need: history>=2 AND has_vectors\n"
                            f"   Current: history={history_len}, vectors={has_vectors}"
                        )
                
                # ============================================================
                # 🔥 BƯỚC 5: LƯU FACE VECTORS VÀO DB (NẾU IDENTIFIED/CONFIRMED)
                # ============================================================
                elif obj_data['status'] in ['identified', 'confirmed']:
                    logger.debug(f"✅ [SKIP IDENTIFY] Track {track_id} already {obj_data['status']}")
                    
                    # 🔥 GỬI SEND-3 KHI STATUS = CONFIRMED
                    if obj_data['status'] == 'confirmed' and obj_data.get('final_attributes'):
                        track_manager.dual_stream_manager.send_stage3_confirmed(
                            person_id=person_id,
                            ai_attributes=obj_data['final_attributes'],
                            llm_sender=track_manager.llm_sender,
                            obj_data=obj_data
                        )
                    # 🔥 LƯU FACE VECTOR VÀO DB NẾU CÒN THIẾU
                    if face_vec is not None and not person_id.startswith('Temp_'):
                        face_count = track_manager.db_manager.count_vectors_for_id(
                            config.FACE_NAMESPACE, 
                            person_id
                        )
                        
                        if face_count < config.MAX_FACE_VECTORS_PER_PROFILE:
                            track_manager.db_manager.add_vectors(
                                config.FACE_NAMESPACE, 
                                person_id, 
                                [face_vec]
                            )
                            logger.info(
                                f"💾 [DB SAVE] Face vector for {person_id} "
                                f"({face_count + 1}/{config.MAX_FACE_VECTORS_PER_PROFILE})"
                            )
                        else:
                            logger.debug(
                                f"⏭️ [SKIP DB] {person_id} already has "
                                f"{face_count}/{config.MAX_FACE_VECTORS_PER_PROFILE} vectors"
                            )
                
                # ============================================================
                # MARK TASK DONE
                # ============================================================
                result_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ [WORKER 3] Process error for track {track_id}: {e}")
                traceback.print_exc()
                result_queue.task_done()
                
        except Exception as e:
            logger.error(f"❌ [WORKER 3] Fatal error: {e}")
            traceback.print_exc()
            time.sleep(0.1)
