"""
CCCD Recognition Worker - Worker 2
Handles face extraction and CCCD (ID card) matching
"""

import queue
import time
import logging
from utils.preprocessing import quick_face_extract

logger = logging.getLogger(__name__)


def fast_cccd_worker(cccd_queue, fast_cccd_instance, llm_sender, dual_stream_manager, track_manager, face_collector, yunet_detector, db_manager):
    """
    WORKER 2 - IMPROVED
    - Handles Pre-Detection (YuNet) asynchronously
    - Handles CCCD Matching
    - NOW WITH db_manager parameter
    
    Args:
        cccd_queue: Queue for CCCD tasks
        fast_cccd_instance: FastCCCDRecognition instance
        llm_sender: LLM sender for notifications
        dual_stream_manager: Dual stream manager
        track_manager: Track manager instance
        face_collector: Face crops collector
        yunet_detector: YuNet face detector
        db_manager: Database manager
    """
    logger.info("🚀 [WORKER 2] FastCCCD Worker started (Extraction + Recognition)")
    
    processed_count = 0
    empty_count = 0
    
    # CCCD Statistics
    cccd_stats = {
        'total_matches': 0,
        'accepted': 0,
        'rejected_mismatch': 0,
        'rejected_no_track': 0
    }
    
    # Cache for visited persons to avoid DB spam
    checked_db_ids = set()
    
    while True:
        try:
            try:
                task = cccd_queue.get(timeout=0.2)
                empty_count = 0
            except queue.Empty:
                empty_count += 1
                if empty_count % 50 == 0: # 10s
                    logger.info(f"💤 [WORKER 2 IDLE] Processed: {processed_count}")
                continue
            
            if task == "STOP":
                break
            
            # ==========================================================
            # TASK TYPE 1: EXTRACT & ACCUMULATE (FROM MAIN LOOP)
            # ==========================================================
            if task.get('type') == 'EXTRACT':
                track_id = task['track_id']
                
                # Check validation
                if not face_collector.should_process_track(track_id):
                    cccd_queue.task_done()
                    continue
                    
                # 1. Run YuNet (Heavy Op)
                face_crop, q_score = quick_face_extract(
                    frame_resized=task['frame_resized'],
                    bbox_resized=task['bbox_resized'],
                    frame_original=task['frame_original'],
                    scale_x=task['scale_x'],
                    scale_y=task['scale_y'],
                    yunet_detector=yunet_detector,
                    min_confidence=0.85,
                    debug_count=task.get('frame_count', 0)
                )
                
                if face_crop is not None:
                    # 2. Get Person ID safely
                    person_id = f'Temp_{track_id}'
                    if track_id in track_manager.tracked_objects:
                        person_id = track_manager.tracked_objects[track_id].get('final_id', f'Temp_{track_id}')

                    # 3. Add to Collector
                    is_ready = face_collector.add_crop(track_id, face_crop, person_id, q_score)
                    
                    if is_ready:
                        data = face_collector.get_crops(track_id)
                        if data:
                            try:
                                cccd_queue.put_nowait({
                                    'type': 'RECOGNIZE',
                                    'track_id': track_id,
                                    'person_id': data['person_id'],
                                    'face_crops': data['crops'],
                                    'avg_quality': data['avg_quality']
                                })
                                logger.info(f"📤 [WORKER 2] Trigger Recognition for {data['person_id']}")
                            except queue.Full:
                                logger.warning(f"⚠️ [WORKER 2] Queue full, dropping RECOGNIZE task")

            # ==========================================================
            # TASK TYPE 2: RECOGNIZE (SELF-DISPATCHED or DIRECT)
            # ==========================================================
            elif task.get('type') == 'RECOGNIZE' or 'face_crops' in task: 
                # Handler for recognition
                person_id = task.get('person_id')
                face_crops = task.get('face_crops', [])
                track_id = task.get('track_id')
                avg_quality = task.get('avg_quality', 0)
                
                processed_count += 1
                
                if avg_quality < 0.4:
                    cccd_queue.task_done()
                    continue

                logger.info(f"🔍 [WORKER 2] RECOGNIZING {person_id} ({len(face_crops)} crops)...")
                
                result = fast_cccd_instance._process_images_improved(
                    track_id=track_id,
                    images=face_crops,
                    person_id=person_id
                )
                
                if result and result.get('matched'):
                    cccd_meta = result['cccd_metadata']
                    cccd_name = cccd_meta.get('name', 'Unknown')
                    
                    cccd_stats['total_matches'] += 1
                    
                    # ===================================================================
                    # VALIDATION: Check person_id matches tracked object
                    # ===================================================================
                    obj_data = track_manager.tracked_objects.get(track_id)
                    
                    if not obj_data:
                        logger.warning(f"⚠️ [CCCD] Track {track_id} not found during update")
                        cccd_queue.task_done()
                        continue
                    
                    actual_person_id = obj_data.get('final_id', f'Temp_{track_id}')
                    
                    if actual_person_id != person_id:
                        cccd_stats['rejected_mismatch'] += 1
                        logger.error(
                            f"[CCCD MISMATCH] Prevented wrong assignment!\n"
                            f"   Task person_id: {person_id}\n"
                            f"   Actual person_id: {actual_person_id}\n"
                            f"   Track ID: {track_id}\n"
                            f"   CCCD: {cccd_name}\n"
                            f"   REJECTED to prevent duplicate assignment"
                        )
                        cccd_queue.task_done()
                        continue
                    
                    # VALIDATION PASSED
                    cccd_stats['accepted'] += 1
                    logger.info(f"   ✅ [MATCH VALID] {person_id} -> {cccd_name}")

                    # PARSE AGE
                    cccd_age_raw = cccd_meta.get('age', 'unknown')
                    cccd_age = 'unknown'
                    if cccd_age_raw and str(cccd_age_raw).lower() != 'unknown':
                        try:
                            cccd_age = int(cccd_age_raw)
                        except:
                            cccd_age = str(cccd_age_raw)
                            
                    # 1. UPDATE MEMORY
                    obj_data['cccd_matched'] = True
                    obj_data['cccd_name'] = cccd_name
                    obj_data['cccd_age'] = cccd_age
                    obj_data['cccd_gender'] = cccd_meta.get('gender', 'unknown')
                    obj_data['cccd_country'] = cccd_meta.get('country', 'unknown')
                    
                    # 🔥 CHUYỂN TEMP → TÊN CCCD (chỉ log 1 lần)
                    if person_id.startswith('Temp_') and obj_data.get('final_id') != cccd_name:
                        obj_data['final_id'] = cccd_name
                        logger.warning(f"🔄 [TEMP→CCCD] {person_id} → {cccd_name}")
                    
                    logger.info(f"🔒 [LOCK CCCD] {person_id} - Set cccd_matched=True to protect from AI overwrite")

                    # 🔥 INSTANT CONFIRM: CCCD match = confirmed immediately!
                    obj_data['status'] = 'confirmed'
                    obj_data['identification_source'] = 'CCCD'
                    logger.info(f"⚡ [INSTANT CONFIRM] {person_id} → CONFIRMED via CCCD match!")

                    # 2. UPDATE FINAL ATTRIBUTES (Ensure dict exists)
                    if not isinstance(obj_data.get('final_attributes'), dict):
                        obj_data['final_attributes'] = {}
                        
                    obj_data['final_attributes'].update({
                        'name': cccd_name,
                        'gender': obj_data['cccd_gender'],
                        'age_onnx': str(cccd_age),
                        'country': obj_data['cccd_country'],
                        'gender_confidence': 1.0,
                        'age_confidence': 1.0
                    })
                    
                    # 3. SAVE TO DATABASE
                    if not person_id.startswith('Temp_'):
                        try:
                            import datetime
                            db_meta = {
                                'cccd_matched': True,
                                'cccd_name': cccd_name,
                                'cccd_age': cccd_age,
                                'cccd_gender': obj_data['cccd_gender'],
                                'cccd_country': obj_data['cccd_country'],
                                'confirmed_name': cccd_name,
                                'confirmed_gender': obj_data['cccd_gender'],
                                'confirmed_age': cccd_age,
                                'confirmed_race': obj_data['cccd_country'],
                                'status': 'confirmed',
                                'last_seen_time': datetime.datetime.now().isoformat()
                            }
                            db_manager.update_metadata(person_id, db_meta)
                            db_manager.save_metadata()
                            logger.info(f"💾 [CCCD SAVED] Saved {person_id} Metadata")
                        except Exception as e:
                            logger.error(f"❌ [DB SAVE ERROR] {person_id}: {e}")

                    logger.info(
                        f"✅ [TRACK UPDATE] {person_id} - CCCD synced: "
                        f"Name={cccd_name}, Age={cccd_age}"
                    )
                    
                    face_collector.mark_person_matched(person_id)
                    
                    dual_stream_manager.on_cccd_result(result)
                    
                    dual_stream_manager.send_stage2_cccd_match(
                        person_id=person_id,
                        llm_sender=llm_sender,
                        obj_data=obj_data
                    )
                    
                    # Log statistics every 10 matches
                    if cccd_stats['total_matches'] % 10 == 0:
                        logger.info(
                            f"[CCCD STATS] Total: {cccd_stats['total_matches']} | "
                            f"Accepted: {cccd_stats['accepted']} | "
                            f"Rejected: {cccd_stats['rejected_mismatch']+cccd_stats['rejected_no_track']} "
                            f"(mismatch:{cccd_stats['rejected_mismatch']}, no_track:{cccd_stats['rejected_no_track']})"
                        )
            
            cccd_queue.task_done()
            
        except Exception as e:
            logger.error(f"❌ [WORKER 2] Error: {e}")
            try: 
                cccd_queue.task_done() 
            except: 
                pass
