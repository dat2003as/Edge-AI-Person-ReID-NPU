"""
Main Processing Loop Module.

Handles the main processing loop including:
- Camera frame reading
- YOLO tracking
- Pre-detection and face collection
- Track management
- Queue cleanup
- ZMQ publishing
- Statistics monitoring
"""

import time
import cv2
import queue
import logging
import json
import zmq
import psutil
import config
from workers.app.helpers import (
    filter_detections_by_confidence,
    prepare_ui_tracked_data,
    prepare_ui_stats,
    cleanup_ghost_tracks,
    filter_tracks_for_ai_processing,
    clear_queue_for_track
)


logger = logging.getLogger(__name__)


def run_main_processing_loop(system_components, profiler, perf_tracker, frame_buffer):
    """
    Main processing loop - handles camera frames, YOLO tracking, face collection, and publishing.
    
    Args:
        system_components: Dict from initialize_system()
        profiler: ResourceProfiler instance
        perf_tracker: PerformanceTracker instance
        frame_buffer: SmartFrameBuffer instance
    """
    # Unpack system components
    zmq_cam = system_components['zmq_cam']
    yolo_model = system_components['yolo_model']
    track_manager = system_components['track_manager']
    face_collector = system_components['face_collector']
    frame_skip_controller = system_components['frame_skip_controller']
    dual_stream_manager = system_components['dual_stream_manager']
    llm_sender = system_components['llm_sender']
    db_manager = system_components['db_manager']
    pub_socket = system_components['pub_socket']
    
    # Queues
    cccd_task_queue = system_components['queues']['cccd']
    attribute_task_queue = system_components['queues']['ai']
    attribute_result_queue = system_components['queues']['result']
    ui_queue = system_components['queues']['ui']
    
    # Quality stats
    reid_quality_stats = system_components['quality_stats']['reid']
    face_quality_stats = system_components['quality_stats']['face']
    
    # Profiler assignment
    profiler_instance = profiler
    track_manager.attributes_manager.profiler = profiler_instance
    track_manager.attributes_manager.reid_face_analyzer.profiler = profiler_instance
    
    # State variables
    frame_count = 0
    is_idle = False
    idle_start_time = None
    YOLO_WIDTH = config.YOLO_WIDTH
    YOLO_HEIGHT = config.YOLO_HEIGHT
    yolo_frame_counter = 0
    last_yolo_results = None
    confirmed_frame_skip = {}
    
    pre_detection_stats = {
        'total_attempts': 0,
        'success_count': 0,
        'sent_to_cccd': 0
    }
    
    last_frame_time = time.time()
    
    logger.info("SYSTEM READY. Press 'Q' to quit.\n")
    
    # Initialize face collector tracking
    if not hasattr(face_collector, 'track_frame_counts'):
        face_collector.track_frame_counts = {}
    
    # MAIN LOOP
    while True:
        profiler_instance.start("Total_Frame")
        
        system_status = "INITIALIZING"
        status_color = (0, 165, 255)
        
        # 1. READ CAMERA
        profiler_instance.start("1. Read_Cam")
        success, frame_original = zmq_cam.read()
        profiler_instance.stop("1. Read_Cam")
        
        if not success or frame_original is None:
            time.sleep(0.01)
            continue
        
        frame_resized = cv2.resize(frame_original, (YOLO_WIDTH, YOLO_HEIGHT))
        
        scale_x = frame_original.shape[1] / YOLO_WIDTH
        scale_y = frame_original.shape[0] / YOLO_HEIGHT
        
        # Debug save every 30 frames
        if frame_count % 30 == 0 and frame_count > 0:
            debug_yolo_path = f"debug_yolo/frame_{frame_count}_{YOLO_WIDTH}x{YOLO_HEIGHT}.jpg"
            cv2.imwrite(debug_yolo_path, frame_resized)
            print(f"[DEBUG] Saved YOLO frame: {debug_yolo_path} | Original: {frame_original.shape[1]}x{frame_original.shape[0]}")
        
        frame_buffer.put(frame_resized)
        
        # Frame rate control
        current_frame_time = time.time()
        time_since_last = current_frame_time - last_frame_time
        
        ai_queue_size = attribute_task_queue.qsize()  
        if ai_queue_size > 10:
            min_interval = 0.2
        elif ai_queue_size > 5:
            min_interval = 0.1
        else:
            min_interval = 0.066
        
        if time_since_last < min_interval:
            continue
        
        last_frame_time = current_frame_time
        frame_count += 1
        yolo_frame_counter += 1
        
        latest_frame = frame_buffer.get_latest()
        if latest_frame is None:
            continue
        
        frame_resized = latest_frame
        
        # 2. YOLO TRACKING
        should_run_yolo = (yolo_frame_counter % (config.YOLO_SKIP_FRAMES + 1) == 0)
        
        if should_run_yolo:
            profiler_instance.start("2. YOLO_Track")
            results = yolo_model.track(
                frame_resized, persist=True, verbose=False,
                conf=0.45, iou=0.4, classes=[0],
                tracker=config.TRACKER_CONFIG_PATH
            )
            last_yolo_results = results
            profiler_instance.stop("2. YOLO_Track")
        else:
            if last_yolo_results is None:
                continue
            results = last_yolo_results
        
        # 3. PUSH UI DATA
        try:
            ui_tracked_data = prepare_ui_tracked_data(track_manager.tracked_objects)
            stats = prepare_ui_stats(
                profiler_instance, 
                frame_buffer, 
                {'ai': attribute_task_queue, 'cccd': cccd_task_queue},
                llm_sender,
                pre_detection_stats
            )
            
            ui_data = {
                'frame': frame_resized.copy(),
                'tracked_objects': ui_tracked_data,
                'stats': stats,
                'system_status': system_status,
                'status_color': status_color,
                'frame_count': frame_count
            }
            
            try:
                ui_queue.put_nowait(ui_data)
            except queue.Full:
                try:
                    ui_queue.get_nowait()
                    ui_queue.put_nowait(ui_data)
                except:
                    pass
        except Exception as e:
            logger.error(f"[UI QUEUE] Error: {e}")
        
        # 4. PARSE DETECTIONS
        track_ids, bboxes, raw_confs = [], [], []
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            raw_confs = results[0].boxes.conf.cpu().numpy()
            
            for box in results[0].boxes.xyxy.cpu().numpy():
                bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        
        clean_ids, clean_bboxes = filter_detections_by_confidence(
            track_ids, bboxes, raw_confs, config
        )
        
        if frame_count % 30 == 0:
            logger.info(f"[DETECTION] Frame {frame_count}: Total={len(track_ids)} | Clean={len(clean_ids)}")
            if len(track_ids) > 0:
                logger.info(f"   Raw IDs: {track_ids}")
                logger.info(f"   Clean IDs: {clean_ids}")
        
        # Queue cleanup
        if frame_count % 10 == 0:
            ai_size = attribute_task_queue.qsize()
            if ai_size > 5:
                frame_skip_controller.cleanup_old_frames(attribute_task_queue)
        
        # 5. HANDLE IDLE STATE
        if not clean_ids:
            if not is_idle:
                is_idle = True
                idle_start_time = time.time()
                
                # Clear queues
                try:
                    while not attribute_task_queue.empty():
                        attribute_task_queue.get_nowait()
                        attribute_task_queue.task_done()
                except:
                    pass
                
                try:
                    while not cccd_task_queue.empty():
                        cccd_task_queue.get_nowait()
                        cccd_task_queue.task_done()
                except:
                    pass
            
            profiler_instance.start("3. Track_Logic")
            track_manager.update_tracks([], [], frame_resized, attribute_task_queue=None)
            profiler_instance.stop("3. Track_Logic")
            
            idle_duration = int(time.time() - idle_start_time) if idle_start_time else 0
            system_status = f"IDLE ({idle_duration}s)"
            status_color = (0, 255, 255)
        
        # 6. HANDLE ACTIVE STATE
        else:
            if is_idle:
                is_idle = False
                idle_duration = time.time() - idle_start_time if idle_start_time else 0
                logger.info(f"[ACTIVE] {len(clean_ids)} people")
                idle_start_time = None
            
            cccd_queue_size = cccd_task_queue.qsize()
            
            if ai_queue_size > 8 or cccd_queue_size > 4:
                continue
            
            # 7. PRE-DETECTION: FACE COLLECTION
            profiler_instance.start("3a. Pre-Detection")
            current_time = time.time()
            
            frame_resized_shared = frame_resized.copy()
            frame_original_shared = frame_original.copy()
            
            for tid, bbox in zip(clean_ids, clean_bboxes):
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                
                if not frame_skip_controller.should_process_frame(
                    track_id=tid,
                    bbox_area=bbox_area,
                    queue_size=ai_queue_size
                ):
                    continue
                
                needs_collection = False
                
                if tid not in face_collector.track_frame_counts:
                    face_collector.track_frame_counts[tid] = 0
                
                if tid not in track_manager.tracked_objects:
                    needs_collection = True
                else:
                    obj = track_manager.tracked_objects[tid]
                    person_id = obj.get('final_id', f'Temp_{tid}')
                    cccd_matched = obj.get('cccd_matched', False)
                    track_frame_count = face_collector.track_frame_counts[tid]
                    
                    if person_id in face_collector.matched_person_ids:
                        needs_collection = False
                    elif track_frame_count >= 200:
                        needs_collection = False
                    elif cccd_matched:
                        needs_collection = False
                    elif face_collector.should_process_track(tid):
                        needs_collection = True
                        face_collector.track_frame_counts[tid] += 1
                        
                        if track_frame_count > 0 and track_frame_count % 50 == 0:
                            crops_count = len(face_collector.collections.get(tid, {}).get('crops', []))
                            logger.info(
                                f"[CCCD PROGRESS] Track {tid} ({person_id}): "
                                f"{track_frame_count}/200 frames, {crops_count} crops collected"
                            )
                
                if needs_collection:
                    pre_detection_stats['total_attempts'] += 1
                    try:
                        if cccd_task_queue.qsize() < 15: 
                            cccd_task_queue.put_nowait({
                                'type': 'EXTRACT',
                                'track_id': tid,
                                'bbox_resized': bbox,
                                'frame_resized': frame_resized_shared,
                                'frame_original': frame_original_shared,
                                'scale_x': scale_x,
                                'scale_y': scale_y,
                                'frame_count': frame_count,
                                'timestamp': current_time
                            })
                            pre_detection_stats['total_attempts'] += 1
                        else:
                            if frame_count % 30 == 0:
                                logger.debug(f"[DROP] CCCD Queue full, skipping extraction for {tid}")
                    except queue.Full:
                        pass
                    except Exception as e:
                        logger.error(f"[DISPATCH ERROR] {e}")
            
            profiler_instance.stop("3a. Pre-Detection")
            
            # 8. TRACK UPDATE
            profiler_instance.start("3b. Track_Logic")
            
            before_count = len(track_manager.tracked_objects)
            
            # Filter: Skip confirmed tracks from AI processing
            ai_track_ids, ai_bboxes = filter_tracks_for_ai_processing(
                track_manager, clean_ids, clean_bboxes
            )
            
            track_manager.update_tracks(
                track_ids=ai_track_ids,
                bboxes=ai_bboxes,
                frame=frame_resized,
                attribute_task_queue=attribute_task_queue,
                frame_original=frame_original,
                scale_x=scale_x,
                scale_y=scale_y
            )
            
            after_count = len(track_manager.tracked_objects)
            logger.info(f"[TRACK UPDATE] After: {after_count} tracks")
            
            profiler_instance.stop("3b. Track_Logic")
            
            # 9. CLEANUP GHOST TRACKS
            profiler_instance.start("3b2. Cleanup_Ghost_Tracks")
            
            current_tracked_ids = set(clean_ids)
            cleanup_ghost_tracks(
                track_manager, current_tracked_ids, confirmed_frame_skip,
                face_collector, frame_skip_controller, current_time
            )
            
            profiler_instance.stop("3b2. Cleanup_Ghost_Tracks")
            
            # 10. PERIODIC CLEANUP
            if frame_count % 50 == 0:
                logger.info(f"[PERIODIC CLEANUP] Frame {frame_count}")
                
                cleared_results = 0
                temp_results = []
                try:
                    while not attribute_result_queue.empty():
                        try:
                            item = attribute_result_queue.get_nowait()
                            temp_results.append(item)
                        except queue.Empty:
                            break
                except:
                    pass
                
                for item in temp_results:
                    try:
                        attribute_result_queue.put_nowait(item)
                    except:
                        cleared_results += 1
                
                if cleared_results > 0:
                    logger.warning(f"[CLEANUP] Dropped {cleared_results} old results")
                
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"[MEMORY] {mem_mb:.0f} MB | Tracked: {len(track_manager.tracked_objects)}")
            
            # Clear old results when track confirmed
            confirmed_tracks = set()
            for track_id, obj_data in track_manager.tracked_objects.items():
                if obj_data.get('status') == 'confirmed':
                    confirmed_tracks.add(track_id)
                    if not obj_data.get('_queue_cleared', False):
                        cleared = clear_queue_for_track(attribute_result_queue, track_id)
                        
                        if cleared > 0:
                            logger.info(f"[QUEUE CLEAR] Track {track_id} confirmed - cleared {cleared} old results")
                        
                        obj_data['_queue_cleared'] = True
            
            # 11. SYNC CCCD DATA
            profiler_instance.start("3c. Process_Results")
            
            for track_id, obj_data in track_manager.tracked_objects.items():
                person_id = obj_data.get('final_id')
                
                if person_id and not person_id.startswith('Temp_'):
                    if obj_data.get('status') == 'confirmed' and obj_data.get('_cccd_synced'):
                        continue
                    
                    try:
                        db_meta = db_manager.get_metadata(person_id)
                        
                        if db_meta and db_meta.get('cccd_matched'):
                            obj_data['cccd_matched'] = True
                            obj_data['cccd_name'] = db_meta.get('cccd_name', 'Unknown')
                            
                            if not isinstance(obj_data.get('final_attributes'), dict):
                                obj_data['final_attributes'] = {}
                            
                            obj_data['final_attributes'].update({
                                'name': db_meta.get('cccd_name', 'Unknown'),
                                'gender': db_meta.get('cccd_gender', 'unknown'),
                                'age_onnx': db_meta.get('cccd_age', 'unknown'),
                                'country': db_meta.get('cccd_country', 'unknown')
                            })
                            
                            if obj_data.get('status') == 'confirmed':
                                obj_data['_cccd_synced'] = True
                    except Exception as e:
                        logger.error(f"[SYNC ERROR] {person_id}: {e}")
                
                elif person_id and person_id.startswith('Temp_'):
                    if obj_data.get('status') == 'confirmed':
                        continue
                    
                    state = dual_stream_manager.get_state(person_id)
                    
                    if state and state.get('cccd_name'):
                        cccd_name = state['cccd_name']
                        
                        logger.warning(
                            f"[TEMP->PERSON] {person_id} matched CCCD: {cccd_name}"
                        )
                        
                        obj_data['status'] = 'confirmed'
                        obj_data['identification_source'] = 'CCCD'
                        obj_data['final_id'] = cccd_name
                        obj_data['cccd_name'] = cccd_name
                        obj_data['cccd_matched'] = True
            
            profiler_instance.stop("3c. Process_Results")
            
            # 12. ZMQ PUBLISH
            if frame_count % 5 == 0:
                profiler_instance.start("3f. ZMQ_Publish")
                
                publish_data = {}
                
                for track_id, obj_data in track_manager.tracked_objects.items():
                    final_id = obj_data.get('final_id', f"Temp_{track_id}")
                    status = obj_data.get('status', 'pending')
                    
                    has_cccd = obj_data.get('cccd_matched', False)
                    db_meta = None
                    
                    if (not has_cccd and db_manager and final_id and 
                        not final_id.startswith('Temp_') and 
                        not obj_data.get('_cccd_synced')):
                        try:
                            db_meta = db_manager.get_metadata(final_id)
                            if db_meta and db_meta.get('cccd_matched'):
                                has_cccd = True
                                obj_data['cccd_matched'] = True
                                obj_data['cccd_name'] = db_meta.get('cccd_name')
                                obj_data['cccd_gender'] = db_meta.get('cccd_gender')
                                obj_data['cccd_age'] = db_meta.get('cccd_age')
                                obj_data['cccd_country'] = db_meta.get('cccd_country')
                        except:
                            pass
                    
                    if final_id.startswith('Temp_'):
                        continue
                    
                    person_info = {
                        'track_id': track_id,
                        'final_id': final_id,
                        'status': status,
                        'bbox': obj_data.get('bbox', [0, 0, 0, 0])
                    }
                    
                    final_attrs = obj_data.get('final_attributes', {}) or {}
                    
                    if has_cccd and obj_data.get('cccd_name'):
                        person_info.update({
                            'name': obj_data.get('cccd_name', 'Unknown'),
                            'gender': obj_data.get('cccd_gender', 'unknown'),
                            'age_onnx': str(obj_data.get('cccd_age', 'unknown')),
                            'country': obj_data.get('cccd_country', 'unknown'),
                            'upper_type': final_attrs.get('upper_type', 'unknown'),
                            'lower_type': final_attrs.get('lower_type', 'unknown'),
                            'emotion': final_attrs.get('emotion', 'unknown'),
                            'source': 'CCCD (memory)',
                            'cccd_matched': True
                        })
                    elif has_cccd and db_meta:
                        try:
                            person_info.update({
                                'name': db_meta.get('cccd_name', 'Unknown'),
                                'gender': db_meta.get('cccd_gender', 'unknown'),
                                'age_onnx': str(db_meta.get('cccd_age', 'unknown')),
                                'country': db_meta.get('cccd_country', 'unknown'),
                                'upper_type': final_attrs.get('upper_type', 'unknown'),
                                'lower_type': final_attrs.get('lower_type', 'unknown'),
                                'emotion': final_attrs.get('emotion', 'unknown'),
                                'source': 'CCCD (DB)',
                                'cccd_matched': True
                            })
                        except Exception as e:
                            logger.error(f"CCCD DB error: {e}")
                            person_info.update({
                                'name': 'Unknown',
                                'gender': final_attrs.get('gender', 'unknown'),
                                'age_onnx': final_attrs.get('age', 'unknown'),
                                'race': final_attrs.get('race', 'unknown'),
                                'upper_type': final_attrs.get('upper_type', 'unknown'),
                                'lower_type': final_attrs.get('lower_type', 'unknown'),
                                'emotion': final_attrs.get('emotion', 'unknown'),
                                'source': 'AI',
                                'cccd_matched': False
                            })
                    else:
                        person_info.update({
                            'name': 'Unknown',
                            'gender': final_attrs.get('gender', 'unknown'),
                            'age_onnx': final_attrs.get('age', 'unknown'),
                            'race': final_attrs.get('race', 'unknown'),
                            'upper_type': final_attrs.get('upper_type', 'unknown'),
                            'lower_type': final_attrs.get('lower_type', 'unknown'),
                            'emotion': final_attrs.get('emotion', 'unknown'),
                            'source': 'AI',
                            'cccd_matched': False
                        })
                    
                    publish_data[str(track_id)] = person_info
                
                if publish_data:
                    try:
                        json_data = json.dumps(publish_data, ensure_ascii=False)
                        
                        pub_socket.send_multipart([
                            b"tracked_data",
                            json_data.encode('utf-8')
                        ], flags=zmq.NOBLOCK)
                        
                        logger.debug(f"[ZMQ] Published {len(publish_data)} persons")
                        
                    except zmq.Again:
                        logger.warning(f"[ZMQ] Send buffer full, skipped frame (web lag?)")
                    except Exception as e:
                        logger.error(f"[ZMQ] Publish error: {e}")
                
                profiler_instance.stop("3f. ZMQ_Publish")
            
            # Cleanup old collections
            if frame_count % 100 == 0:
                active_ids = set(track_manager.tracked_objects.keys())
                face_collector.cleanup_old(active_ids)
                frame_skip_controller.cleanup_old_timestamps(active_ids)
            
            system_status = f"ACTIVE ({len(clean_ids)})"
            status_color = (0, 255, 0)
        
        # 13. MONITOR STATISTICS
        if frame_count % 100 == 0 and frame_count > 0:
            logger.info("\n" + "="*60)
            logger.info("[QUALITY STATISTICS REPORT]")
            logger.info("="*60)
            
            total_reid = reid_quality_stats['total']
            if total_reid > 0:
                accept_rate_reid = (reid_quality_stats['accepted'] / total_reid) * 100
                logger.info(
                    f"\n[REID BODY QUALITY]\n"
                    f"   Total processed: {total_reid}\n"
                    f"   Accepted: {reid_quality_stats['accepted']} ({accept_rate_reid:.1f}%)\n"
                    f"   Rejected:\n"
                    f"      - Blur: {reid_quality_stats['rejected_blur']}\n"
                    f"      - Dark: {reid_quality_stats['rejected_dark']}\n"
                    f"      - Low contrast: {reid_quality_stats['rejected_contrast']}\n"
                    f"      - Too small: {reid_quality_stats['rejected_small']}\n"
                )
            
            total_face = face_quality_stats['total']
            if total_face > 0:
                accept_rate_face = (face_quality_stats['accepted'] / total_face) * 100
                logger.info(
                    f"\n[FACE QUALITY]\n"
                    f"   Total processed: {total_face}\n"
                    f"   Accepted: {face_quality_stats['accepted']} ({accept_rate_face:.1f}%)\n"
                    f"   Rejected:\n"
                    f"      - Blur: {face_quality_stats['rejected_blur']}\n"
                    f"      - Low conf: {face_quality_stats['rejected_low_conf']}\n"
                )
            
            logger.info("="*60 + "\n")
            
            if frame_count % 500 == 0:
                reid_quality_stats = {k: 0 for k in reid_quality_stats}
                face_quality_stats = {k: 0 for k in face_quality_stats}
                logger.info("[RESET] Quality stats reset for next 500 frames\n")
        
        profiler_instance.stop("Total_Frame")
        
        # 14. PERFORMANCE TRACKING
        stats_data, _, _ = profiler_instance.get_stats()
        frame_duration_ms = stats_data.get("Total_Frame", 0)
        
        perf_tracker.record_frame(
            frame_duration=frame_duration_ms,
            track_manager=track_manager,
            ai_queue_size=attribute_task_queue.qsize(),
            cccd_queue_size=cccd_task_queue.qsize(),
            result_queue_size=attribute_result_queue.qsize()
        )
        
        if perf_tracker.should_report():
            perf_tracker.print_report()
        
        if frame_count % 30 == 0:
            profiler_instance.print_report()
