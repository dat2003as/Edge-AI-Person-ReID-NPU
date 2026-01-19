"""
Main application entry point - refactored for modularity.

This is the entry point for the person tracking system.
The implementation has been refactored into modular components:
- initializer.py: System initialization
- ui_renderer.py: UI rendering thread
- main_loop.py: Main processing loop
- helpers.py: Helper utilities

This preserves all original functionality while improving code organization.
"""

import logging
import traceback
import cv2
import sys

from utils.profiler import ResourceProfiler
from utils.performance_tracker import PerformanceTracker
from core.collectors import SmartFrameBuffer
from workers.app.initializer import initialize_system
from workers.app.ui_renderer import start_ui_rendering_thread
from workers.app.main_loop import run_main_processing_loop

logger = logging.getLogger(__name__)


def run_application():
    """
    Main application entry point - orchestrates the system components.
    """
    print("=== HỆ THỐNG 3 WORKERS: AI + CCCD + RESULT ===\n")
    
    # Initialize performance tracking
    profiler = ResourceProfiler()
    perf_tracker = PerformanceTracker(window_size=30, report_interval=1)
    frame_buffer = SmartFrameBuffer(max_size=10)
    
    try:
        # 1. Initialize all system components
        logger.info("Initializing system components...")
        system_components = initialize_system()
        logger.info("System initialization complete\n")
        
        # 2. Start UI rendering thread
        ui_thread = start_ui_rendering_thread(
            system_components['queues']['ui'],
            system_components['track_manager']
        )
        logger.info("UI thread started\n")
        
        logger.info("SYSTEM READY. Press 'Q' to quit.\n")
        
        # 3. Run main processing loop
        run_main_processing_loop(system_components, profiler, perf_tracker, frame_buffer)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    
    finally:
        logger.info("\nCleaning up...")
        try:
            if 'system_components' in locals() and system_components:
                system_components['zmq_cam'].release()
        except Exception as e:
            logger.error(f"Error releasing ZMQ camera: {e}")
        cv2.destroyAllWindows()
        logger.info("Done")
