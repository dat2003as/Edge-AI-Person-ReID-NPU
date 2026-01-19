"""
ZMQ Camera interface for receiving frames from remote camera server.
"""

import zmq
import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class ZMQCamera:
    """
    ZeroMQ-based camera client for receiving frames from remote server.
    Supports automatic reconnection and frame timeout handling.
    """
    
    def __init__(self, server_url, max_retries=5, retry_delay=2.0):
        self.server_url = server_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.context = None
        self.socket = None
        self.connected = False
        self.retry_count = 0
        self.last_frame_time = None
        self.frame_timeout = 10.0  # Increased from 5.0s to avoid reconnect during heavy processing
        self.first_frame_received = False
        
        self._connect()
    
    def _connect(self):
        """Establish connection to ZMQ server with retry logic."""
        try:
            if self.socket is not None:
                try:
                    self.socket.close()
                except:
                    pass
            
            if self.context is not None:
                try:
                    self.context.term()
                except:
                    pass
            
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            
            self.socket.setsockopt(zmq.RCVHWM, 1)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(self.server_url)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.socket.setsockopt(zmq.CONFLATE, 1)
            
            self.connected = True
            self.retry_count = 0
            self.last_frame_time = time.time()
            self.first_frame_received = False
            logger.info(f"✅ Connected to camera: {self.server_url}")
            
        except Exception as e:
            self.connected = False
            logger.error(f"❌ Connection failed: {e}")
            self.retry_count += 1
            
            if self.retry_count < self.max_retries:
                logger.warning(f"⏳ Retry {self.retry_count}/{self.max_retries} in {self.retry_delay}s...")
                time.sleep(self.retry_delay)
                self._connect()
            else:
                logger.error(f"❌ Max retries ({self.max_retries}) exceeded!")
    
    def read(self):
        """
        Read a frame from ZMQ camera.
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        try:
            if self.first_frame_received:
                current_time = time.time()
                if current_time - self.last_frame_time > self.frame_timeout:
                    logger.warning(f"⏱️ Frame timeout ({self.frame_timeout}s) - Reconnecting...")
                    self._connect()
                    return False, None
            
            if self.socket.poll(timeout=200):
                raw_image = self.socket.recv()
                img_array = np.frombuffer(raw_image, dtype=np.uint8)
                frame = cv2.imdecode(
                    img_array, 
                    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )
                
                if frame is not None:
                    current_time = time.time()
                    self.last_frame_time = current_time
                    
                    if not self.first_frame_received:
                        self.first_frame_received = True
                        logger.info(f"✅ First frame received! ({frame.shape[1]}x{frame.shape[0]})")
                    
                    return True, frame
                else:
                    logger.warning("⚠️ Failed to decode image")
                    return False, None
            
            return False, None
            
        except zmq.error.Again:
            return False, None
        
        except zmq.error.ContextTerminated:
            logger.warning("⚠️ Context terminated - Reconnecting...")
            self._connect()
            return False, None
        
        except Exception as e:
            logger.error(f"❌ Read error: {e}")
            
            if not self.connected or self.retry_count >= self.max_retries:
                self._connect()
            
            return False, None

    def isOpened(self):
        """Check if camera connection is active."""
        return self.connected and self.socket is not None

    def release(self):
        """Release camera resources and close connection."""
        try:
            if self.socket is not None:
                self.socket.close()
            if self.context is not None:
                self.context.term()
            self.connected = False
            logger.info("✅ Camera closed")
        except Exception as e:
            logger.error(f"❌ Error closing camera: {e}")
