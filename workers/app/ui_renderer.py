"""
UI Rendering Thread for the main application.

Handles the dedicated UI rendering thread that displays:
- Tracked objects visualization
- System statistics
- Performance metrics
"""

import threading
import queue
import time
import cv2
import logging
import os
from draw import draw_tracked_objects

logger = logging.getLogger(__name__)


def start_ui_rendering_thread(ui_queue, track_manager):
    """
    Start the dedicated UI rendering thread.
    
    Args:
        ui_queue: Queue containing UI data to render
        track_manager: TrackManager instance for accessing tracked objects
        
    Returns:
        threading.Thread: Started UI thread instance
    """
    
    def ui_rendering_thread():
        """Dedicated thread for UI rendering"""
        logger.info("[UI THREAD] Started")
        
        while True:
            try:
                ui_data = None
                while not ui_queue.empty():
                    try:
                        ui_data = ui_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if ui_data is None:
                    time.sleep(0.01)
                    continue
                    
                frame_resized = ui_data['frame']
                tracked_objects = ui_data['tracked_objects']
                stats = ui_data['stats']
                system_status = ui_data['system_status']
                status_color = ui_data['status_color']
                frame_count = ui_data['frame_count']
                
                vis_frame = draw_tracked_objects(
                    frame_resized.copy(), 
                    tracked_objects,
                    db_manager=track_manager.db_manager
                )
                
                # Optional: Uncomment to show system stats overlay
                # cv2.rectangle(vis_frame, (5, 5), (285, 165), (0, 0, 0), -1)
                # cv2.putText(vis_frame, f"SYS: {system_status[:15]}", (10, 25), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                # cv2.putText(vis_frame, f"FPS:{stats['fps_est']:.1f} CPU:{stats['cpu']:.0f}%", 
                #            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # cv2.putText(vis_frame, f"RAM:{stats['ram']:.0f}MB Q:{stats['ai_q']}/{stats['cccd_q']}", 
                #            (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # if stats['face_attempts'] > 0:
                #     success_rate = (stats['face_success'] / stats['face_attempts']) * 100
                #     cv2.putText(vis_frame, f"Face:{stats['face_success']}/{stats['face_attempts']} ({success_rate:.0f}%)", 
                #                (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # llm_total = stats['llm_sent'] + stats['llm_failed']
                # success_rate_llm = (stats['llm_sent'] / llm_total * 100) if llm_total > 0 else 0
                # llm_color = (0, 255, 0) if success_rate_llm > 90 else (0, 165, 255)
                # cv2.putText(vis_frame, f"LLM: {stats['llm_sent']}S/{stats['llm_failed']}F ({success_rate_llm:.0f}%)", 
                #            (10, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.5, llm_color, 1)
                # cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 140), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                
                # latency_ms = stats['latency'] * 1000
                # cv2.putText(vis_frame, f"Camera: OK | Latency: {latency_ms:.0f}ms", 
                #            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
               
                cv2.imshow("System", vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logger.info("[UI THREAD] Quit")
                    os._exit(0)
                    
            except Exception as e:
                logger.error(f"[UI THREAD] Error: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("[UI THREAD] Stopped")
    
    ui_thread = threading.Thread(
        target=ui_rendering_thread, 
        daemon=True, 
        name="UI-Rendering"
    )
    ui_thread.start()
    
    return ui_thread
