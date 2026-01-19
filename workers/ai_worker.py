"""
AI Analysis Worker - Worker 1
Processes frames for AI attribute analysis (age, gender, race, clothing, etc.)
"""

import asyncio
import time
import queue
import logging
import traceback

logger = logging.getLogger(__name__)


def combined_analysis_worker(task_queue, result_queue, analyzer, track_manager, face_collector, cccd_queue):
    """
    🔥 WORKER WITH STALE TASK FILTERING
    
    Args:
        task_queue: Input queue for analysis tasks
        result_queue: Output queue for results
        analyzer: Analyzer instance
        track_manager: TrackManager instance
        face_collector: FaceCropsCollector instance
        cccd_queue: CCCD task queue
    """
    logger.info("[WORKER 1] AI Worker starting...")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    processed_count = 0
    skipped_stale = 0
    iteration = 0
    empty_count = 0
    MAX_TASK_AGE = 5  # 🔥 Chỉ xử lý tasks trong 5 giây gần đây
    
    try:
        while True:
            iteration += 1
            
            # Log worker health (reduced from 100 to 300 iterations)
            if iteration % 50 == 0:
                q_size = task_queue.qsize()
                # logger.info(f"[WORKER ALIVE] Iter {iteration} | Queue: {q_size} | Processed: {processed_count} | Skipped: {skipped_stale}")
            
            try:
                # 🔥 GET SINGLE TASK
                try:
                    task = task_queue.get(timeout=0.05)
                    empty_count = 0
                except queue.Empty:
                    empty_count += 1
                    
                    if empty_count % 200 == 0:
                        logger.warning(f"[WORKER STARVING] No tasks for {empty_count * 0.05:.1f}s")
                    
                    continue
                
                # 🔥 VALIDATE TASK
                if not isinstance(task, dict):
                    logger.warning(f"⚠️ [INVALID TASK] Type: {type(task)}")
                    task_queue.task_done()
                    continue
                
                # 🔥 CHECK TASK AGE - SKIP IF TOO OLD
                task_timestamp = task.get('timestamp', 0)
                task_age = time.time() - task_timestamp
                
                if task_age > MAX_TASK_AGE:
                    skipped_stale += 1
                    track_id = task.get('track_id', 'unknown')
                    
                    if skipped_stale % 10 == 0:  # Log mỗi 10 tasks cũ
                        logger.warning(f"⏭️ [SKIP STALE] Track {track_id} - Age: {task_age:.1f}s > {MAX_TASK_AGE}s")
                    
                    task_queue.task_done()
                    continue

                person_id = task.get('person_id', 'unknown')
                track_id = task.get('track_id')

                logger.info(f"[WORKER] Received task for Track {track_id} (Age: {task_age:.2f}s)")
                
                # Check track còn tồn tại
                if track_id not in track_manager.tracked_objects:
                    logger.debug(f"⏭️ [SKIP] Track {track_id} not found")
                    task_queue.task_done()
                    continue
                
                # 🔥 LẤY CẢ 2 FRAMES (backward compatible với task cũ)
                frame_resized = task.get('frame_resized', task.get('frame'))  # 640x480
                frame_original = task.get('frame_original', frame_resized)     # 2K (fallback)
                bbox_resized = task.get('bbox_resized', task.get('bbox'))
                bbox_original = task.get('bbox_original', bbox_resized)
                
                # Analyze với CẢ 2 FRAMES
                try:
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            analyzer.analyze_person_by_bbox(
                                frame_resized=frame_resized,      # 640x480 cho AI analysis
                                frame_original=frame_original,    # 2K cho face extraction
                                bbox_resized=bbox_resized,        # bbox 640x480
                                bbox_original=bbox_original,      # bbox 2K
                                person_id=task['person_id'],
                                confirmed_status=task.get('confirmed_status', 'pending')  # 🔥 2-STAGE PROCESSING
                            ),
                            timeout=2.0
                        )
                    )
                    
                    if result:
                        result_queue.put((track_id, result))
                        processed_count += 1
                        logger.info(f"   ✅ Processed Track {track_id} (Total: #{processed_count})")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ [TIMEOUT] Track {track_id}")
                except Exception as e:
                    logger.error(f"❌ [ANALYZE ERROR] Track {track_id}: {str(e)[:100]}")
                
                task_queue.task_done()
                
            except Exception as e:
                logger.error(f"[WORKER ERROR] {e}")
                traceback.print_exc()
                time.sleep(0.1)
    
    finally:
        logger.info("🛑 [WORKER 1] Shutting down...")
        try:
            loop.close()
        except:
            pass
