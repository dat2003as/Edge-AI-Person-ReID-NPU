"""
Frame skip controller for managing frame processing rates.
"""

import time
import queue
import threading
import logging

logger = logging.getLogger(__name__)


class FrameSkipController:
    """
    Controller quản lý việc skip frames để tránh queue bị tràn
    """
    
    def __init__(self, max_age_ms=200):
        self.max_age_ms = max_age_ms
        self.last_frame_time = {}
        self.frame_timestamps = {}
        self.lock = threading.Lock()
        self.frames_skipped = 0
        self.frames_processed = 0
    
    def should_process_frame(self, track_id, bbox_area, queue_size):
        """
        Determine if a frame should be processed based on bbox size and queue status.
        
        Args:
            track_id: Track ID
            bbox_area: Bounding box area
            queue_size: Current queue size
            
        Returns:
            bool: True if frame should be processed
        """
        with self.lock:
            current_time = time.time()
            if bbox_area > 60000:
                # Người lớn → LUÔN PROCESS (không skip)
                self.last_frame_time[track_id] = current_time
                self.frames_processed += 1
                return True

            if queue_size > 15:
                self.frames_skipped += 1
                return False
            
            if bbox_area < 20000:
                self.frames_skipped += 1
                return False
            
            if track_id in self.last_frame_time:
                time_since_last = current_time - self.last_frame_time[track_id]
                
                if bbox_area > 80000:
                    min_interval = 0.1
                elif bbox_area > 50000:
                    min_interval = 0.15
                elif bbox_area > 30000:
                    min_interval = 0.2
                else:
                    min_interval = 0.3
                
                if time_since_last < min_interval:
                    self.frames_skipped += 1
                    return False
            
            self.last_frame_time[track_id] = current_time
            self.frame_timestamps[track_id] = current_time
            self.frames_processed += 1
            return True
    
    def cleanup_old_frames(self, attribute_task_queue):
        """
        Remove stale frames from the queue.
        
        Args:
            attribute_task_queue: Queue to clean up
            
        Returns:
            int: Number of frames cleared
        """
        current_time = time.time()
        cleared = 0
        kept_tasks = []
        
        try:
            while not attribute_task_queue.empty():
                try:
                    task = attribute_task_queue.get_nowait()
                    
                    if not isinstance(task, dict):
                        attribute_task_queue.task_done()
                        continue
                    
                    task_age = current_time - task.get('timestamp', 0)
                    
                    if task_age < 1.0:
                        kept_tasks.append(task)
                    else:
                        cleared += 1
                        attribute_task_queue.task_done()
                        
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        for task in kept_tasks:
            try:
                attribute_task_queue.put_nowait(task)
            except queue.Full:
                logger.warning("⚠️ Queue full during cleanup")
                break
        
        if cleared > 0:
            logger.info(f"🗑️ [Cleanup] Removed {cleared} stale frames")
        
        return cleared
    
    def cleanup_old_timestamps(self, active_track_ids):
        """
        Clean up timestamps for inactive tracks.
        
        Args:
            active_track_ids: Set of currently active track IDs
        """
        with self.lock:
            inactive_ids = set(self.last_frame_time.keys()) - active_track_ids
            for tid in inactive_ids:
                del self.last_frame_time[tid]
            
            inactive_ts = set(self.frame_timestamps.keys()) - active_track_ids
            for tid in inactive_ts:
                del self.frame_timestamps[tid]
    
    def get_stats(self):
        """
        Get processing statistics.
        
        Returns:
            dict: Stats containing processed, skipped, skip_rate, and total
        """
        with self.lock:
            total = self.frames_processed + self.frames_skipped
            skip_rate = (self.frames_skipped / max(total, 1)) * 100
            
            return {
                'processed': self.frames_processed,
                'skipped': self.frames_skipped,
                'skip_rate': skip_rate,
                'total': total
            }
    
    def reset_stats(self):
        """Reset statistics counters."""
        with self.lock:
            self.frames_processed = 0
            self.frames_skipped = 0
