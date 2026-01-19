"""
Helper functions for the main application.

Contains utility functions for:
- Queue management
- Bbox filtering
- UI data preparation
- Cleanup operations
"""

import queue
import logging
import time

logger = logging.getLogger(__name__)


def clear_queue(queue_obj):
    """
    Clear all items from a queue.
    
    Args:
        queue_obj: Queue to clear
        
    Returns:
        int: Number of items cleared
    """
    cleared = 0
    try:
        while not queue_obj.empty():
            try:
                queue_obj.get_nowait()
                queue_obj.task_done()
                cleared += 1
            except queue.Empty:
                break
    except Exception as e:
        logger.error(f"Error clearing queue: {e}")
    
    return cleared


def clear_queue_for_track(result_queue, track_id):
    """
    Clear all results from queue for a specific track ID.
    
    Args:
        result_queue: Queue containing results
        track_id: Track ID to clear results for
        
    Returns:
        int: Number of items cleared
    """
    cleared = 0
    temp_results = []
    
    try:
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                result_track_id = result.get('track_id')
                if result_track_id != track_id:
                    temp_results.append(result)
                else:
                    cleared += 1
            except queue.Empty:
                break
    except Exception as e:
        logger.error(f"Error clearing queue for track {track_id}: {e}")
    
    for result in temp_results:
        try:
            result_queue.put_nowait(result)
        except queue.Full:
            pass
    
    return cleared


def filter_detections_by_confidence(track_ids, bboxes, raw_confs, config):
    """
    Filter detections based on area-adaptive confidence thresholds.
    
    Args:
        track_ids: List of track IDs
        bboxes: List of bboxes [x1, y1, x2, y2]
        raw_confs: Raw confidence scores
        config: Config module with MIN_BBOX_AREA, MAX_BBOX_AREA
        
    Returns:
        tuple: (clean_ids, clean_bboxes)
    """
    clean_ids, clean_bboxes = [], []
    
    for tid, bbox, conf in zip(track_ids, bboxes, raw_confs):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Area-adaptive confidence threshold
        if area > 200000:
            min_conf = 0.70
        elif area > 80000:
            min_conf = 0.40
        elif area > 40000:
            min_conf = 0.45
        else:
            min_conf = 0.50
        
        if conf >= min_conf and config.MIN_BBOX_AREA < area < config.MAX_BBOX_AREA:
            clean_ids.append(tid)
            clean_bboxes.append(bbox)
        else:
            logger.debug(
                f"[FILTER] Track {tid} | "
                f"Area: {area:.0f} | Conf: {conf:.2f} < {min_conf:.2f}"
            )
    
    return clean_ids, clean_bboxes


def prepare_ui_tracked_data(tracked_objects):
    """
    Prepare tracked objects data for UI rendering.
    
    Args:
        tracked_objects: Dict of tracked objects from TrackManager
        
    Returns:
        dict: Simplified UI-friendly tracked data
    """
    ui_tracked_data = {}
    
    for tid, obj in tracked_objects.items():
        history = obj.get('history_attributes', [])
        last_history = [history[-1]] if history else []
        
        ui_tracked_data[tid] = {
            'bbox': obj.get('bbox'),
            'status': obj.get('status'),
            'final_id': obj.get('final_id'),
            'final_attributes': obj.get('final_attributes'),
            'cccd_name': obj.get('cccd_name'),
            'cccd_matched': obj.get('cccd_matched'),
            'identification_source': obj.get('identification_source'),
            'history_attributes': last_history,
            'last_keypoints': obj.get('last_keypoints')
        }
    
    return ui_tracked_data


def prepare_ui_stats(profiler, frame_buffer, queues, llm_sender, pre_detection_stats):
    """
    Prepare statistics for UI display.
    
    Args:
        profiler: ResourceProfiler instance
        frame_buffer: SmartFrameBuffer instance
        queues: Dict with 'ai', 'cccd' queue keys
        llm_sender: LLMSender instance
        pre_detection_stats: Dict with face detection stats
        
    Returns:
        dict: UI statistics
    """
    stats_data, cpu_p, mem_mb = profiler.get_stats()
    fps_est = 1000 / (stats_data.get("Total_Frame", 1) + 1e-5)
    buffer_stats = frame_buffer.get_stats()
    
    return {
        'fps_est': fps_est,
        'cpu': cpu_p,
        'ram': mem_mb,
        'ai_q': queues['ai'].qsize(),
        'cccd_q': queues['cccd'].qsize(),
        'face_attempts': pre_detection_stats['total_attempts'],
        'face_success': pre_detection_stats['success_count'],
        'llm_sent': llm_sender.total_sent,
        'llm_failed': llm_sender.total_failed,
        'latency': buffer_stats['latency_estimate']
    }


def cleanup_old_results(result_queue, max_items=None):
    """
    Cleanup old results from queue.
    
    Args:
        result_queue: Queue to cleanup
        max_items: Maximum items to keep (None = keep all)
        
    Returns:
        int: Number of items cleared
    """
    cleared_results = 0
    temp_results = []
    
    try:
        while not result_queue.empty():
            try:
                item = result_queue.get_nowait()
                temp_results.append(item)
            except queue.Empty:
                break
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    for item in temp_results:
        try:
            result_queue.put_nowait(item)
        except queue.Full:
            cleared_results += 1
    
    return cleared_results


def cleanup_ghost_tracks(track_manager, current_tracked_ids, confirmed_frame_skip, 
                        face_collector, frame_skip_controller, current_time):
    """
    Remove ghost tracks that are no longer detected.
    
    Args:
        track_manager: TrackManager instance
        current_tracked_ids: Set of currently detected track IDs
        confirmed_frame_skip: Dict tracking confirmed tracks
        face_collector: FaceCropsCollector instance
        frame_skip_controller: FrameSkipController instance
        current_time: Current timestamp
        
    Returns:
        list: List of removed ghost track IDs
    """
    all_tracked_ids = set(track_manager.tracked_objects.keys())
    ghost_ids = all_tracked_ids - current_tracked_ids
    removed_ids = []
    
    for ghost_id in ghost_ids:
        obj_data = track_manager.tracked_objects[ghost_id]
        
        if 'ghost_start_time' not in obj_data:
            obj_data['ghost_start_time'] = current_time
        
        ghost_duration = current_time - obj_data['ghost_start_time']
        
        if ghost_duration > 0.01:
            person_id = obj_data.get('final_id', f'Temp_{ghost_id}')
            logger.info(f"[REMOVE] Track {ghost_id} ({person_id}) - Ghost for {ghost_duration:.2f}s")
            
            del track_manager.tracked_objects[ghost_id]
            removed_ids.append(ghost_id)
            
            if ghost_id in confirmed_frame_skip:
                del confirmed_frame_skip[ghost_id]
            
            if ghost_id in face_collector.collections:
                del face_collector.collections[ghost_id]
            
            if ghost_id in frame_skip_controller.last_frame_time:
                del frame_skip_controller.last_frame_time[ghost_id]
            if ghost_id in frame_skip_controller.frame_timestamps:
                del frame_skip_controller.frame_timestamps[ghost_id]
    
    return removed_ids


def filter_tracks_for_ai_processing(track_manager, clean_ids, clean_bboxes):
    """
    Filter tracks: skip confirmed tracks, only process pending ones.
    
    Args:
        track_manager: TrackManager instance
        clean_ids: List of detected track IDs
        clean_bboxes: List of detected bboxes
        
    Returns:
        tuple: (ai_track_ids, ai_bboxes) - filtered for AI processing
    """
    ai_track_ids = []
    ai_bboxes = []
    
    for tid, bbox in zip(clean_ids, clean_bboxes):
        obj_data = track_manager.tracked_objects.get(tid)
        if obj_data and obj_data.get('status') == 'confirmed':
            obj_data['bbox'] = bbox
            obj_data['last_seen'] = time.time()
            continue
        
        ai_track_ids.append(tid)
        ai_bboxes.append(bbox)
    
    return ai_track_ids, ai_bboxes
