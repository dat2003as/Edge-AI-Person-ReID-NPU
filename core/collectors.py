"""
Collectors for face crops and frame buffering.
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)


class FaceCropsCollector:
    """
    Collector frame mặt để gửi CCCD: 5 frame gửi 1 lần, max 200 frame (20 lần) → dừng
    """
    
    def __init__(self, batch_size=5, max_frames=200, min_quality=0.25, db_manager=None):
        self.batch_size = batch_size
        self.max_frames = max_frames  # Max 200 frame / track_id
        self.min_quality = min_quality
        self.collections = {}
        self.db_manager = db_manager
        self.lock = threading.Lock()
        self.matched_person_ids = set()
        self.frame_counter = {}
    
    def add_crop(self, track_id, face_crop, person_id, q_score=0.0):
        """
        🔥 SIMPLE - CHỈ check confidence, bỏ diversity
        
        Args:
            track_id: Track ID
            face_crop: Face crop image
            person_id: Person ID
            q_score: Quality score
            
        Returns:
            bool: True if ready to send batch
        """
        with self.lock:
            # 🔥 Bỏ qua nếu đã MATCHED CCCD (dừng ngay)
            if person_id in self.matched_person_ids:
                return False
            
            # 🔥 SKIP NẾU ĐÃ CÓ NAME TRONG DB (CCCD đã match rồi)
            if not person_id.startswith('Temp_'):
                try:
                    if self.db_manager:
                        db_meta = self.db_manager.get_metadata(person_id)
                        if db_meta and db_meta.get('cccd_matched'):
                            logger.info(
                                f"⏭️ [SKIP CCCD] {person_id} - "
                                f"Already has CCCD name: {db_meta.get('cccd_name')}"
                            )
                            # 🔥 MARK matched để không check lại
                            self.matched_person_ids.add(person_id)
                            return False
                    else:
                        logger.warning("⚠️ [COLLECTOR] db_manager not set!")
                except Exception as e:
                    logger.error(f"❌ [COLLECTOR DB CHECK] {person_id}: {e}")
            
            # Khởi tạo collection
            if track_id not in self.collections:
                self.collections[track_id] = {
                    'crops': [],
                    'quality_scores': [],
                    'person_id': person_id,
                    'frame_count': 0,
                    'batches_sent': 0,
                    'is_done': False,
                    'total_quality': 0.0
                }
            
            col = self.collections[track_id]
            
            if col['is_done']:
                return False
            
            col['person_id'] = person_id
            col['frame_count'] += 1
            
            # 🔥 LIMIT 200 FRAMES - Sau 200 frame, dừng collect CCCD
            MAX_CCCD_FRAMES = 200
            if col['frame_count'] > MAX_CCCD_FRAMES:
                col['is_done'] = True
                logger.warning(f"⏹️ [CCCD TIMEOUT] Track {track_id} ({person_id}) - Exceeded {MAX_CCCD_FRAMES} frames")
                return False
            
            # 🔥 CHỈ CHECK CONFIDENCE
            if q_score < self.min_quality:
                return False
            
            # 🔥 THU THẬP CROPS (không check diversity)
            if len(col['crops']) < self.batch_size:
                col['crops'].append(face_crop.copy())
                col['quality_scores'].append(q_score)
                col['total_quality'] += q_score
                
                logger.info(
                    f"✅ [COLLECT] Track {track_id} | "
                    f"Person: {person_id} | "
                    f"Crop #{len(col['crops'])}/{self.batch_size} | "
                    f"Conf: {q_score:.3f} | "
                    f"Frame: {col['frame_count']}/{self.max_frames}"
                )
            
            # 🔥 CHECK READY TO SEND
            if len(col['crops']) >= 3:
                return True  # Ít nhất 3 crops
            
            return False
    
    def get_crops(self, track_id):
        """
        🔥 SIMPLE - Lấy crops theo thứ tự, không sort
        
        Args:
            track_id: Track ID
            
        Returns:
            dict: Crop data or None
        """
        with self.lock:
            if track_id not in self.collections:
                return None
            
            col = self.collections[track_id]
            
            if not col['crops']:
                return None
            
            # 🔥 LẤY TẤT CẢ CROPS (không sort, giữ nguyên thứ tự)
            crops_to_send = col['crops'][:self.batch_size]
            avg_q = col['total_quality'] / len(col['crops']) if col['crops'] else 0
            
            data_to_send = {
                'crops': crops_to_send,
                'person_id': col['person_id'],
                'timestamp': time.time(),
                'batch_index': col['batches_sent'] + 1,
                'avg_quality': avg_q
            }
            
            # Reset
            col['crops'] = []
            col['quality_scores'] = []
            col['total_quality'] = 0.0
            col['batches_sent'] += 1
            
            logger.info(
                f"📤 [GET CROPS] Track {track_id} - "
                f"{len(crops_to_send)} crops | "
                f"AvgConf: {avg_q:.3f}"
            )
            
            return data_to_send
    
    def should_process_track(self, track_id):
        """Check if track should continue processing."""
        with self.lock:
            if track_id not in self.collections:
                return True
            col = self.collections[track_id]
            # 🔥 Dừng nếu person đã matched CCCD
            if col['person_id'] in self.matched_person_ids:
                return False
            return not col['is_done']
    
    def mark_person_matched(self, person_id):
        """🔥 Callback khi CCCD match thành công - dừng collection ngay"""
        with self.lock:
            self.matched_person_ids.add(person_id)
            logger.info(f"🛑 [COLLECTOR] Person {person_id} matched CCCD - stopping collection")
            
            # Mark tất cả collections của person này là done
            for track_id, col in self.collections.items():
                if col['person_id'] == person_id:
                    col['is_done'] = True
                    logger.debug(f"   → Track {track_id} marked as done")
    
    def cleanup_old(self, active_track_ids):
        """Clean up collections for inactive tracks."""
        with self.lock:
            inactive = set(self.collections.keys()) - active_track_ids
            for tid in inactive:
                del self.collections[tid]
                logger.debug(f"🧹 [CLEANUP] Removed collection for track {tid}")


class SmartFrameBuffer:
    """
    Smart frame buffer that keeps only the latest frames.
    """
    
    def __init__(self, max_size=1):
        self.max_size = max_size
        self.buffer = []
        self.lock = threading.Lock()
        self.frame_count = 0
        self.dropped_frames = 0

    def put(self, frame):
        """Add frame to buffer, dropping old ones if full."""
        with self.lock:
            self.frame_count += 1
            
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)
                self.dropped_frames += 1
            
            self.buffer.append({
                'frame': frame,
                'timestamp': time.time()
            })

    def get_latest(self):
        """Get the latest frame from buffer."""
        with self.lock:
            if not self.buffer:
                return None
            
            latest = self.buffer[-1]
            return latest['frame']

    def get_stats(self):
        """Get buffer statistics."""
        with self.lock:
            latency = 0
            if self.buffer:
                latency = time.time() - self.buffer[-1]['timestamp']
            
            return {
                'size': len(self.buffer),
                'dropped': self.dropped_frames,
                'total_received': self.frame_count,
                'latency_estimate': latency
            }
