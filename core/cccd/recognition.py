# core/cccd/recognition.py - COMPLETE FIXED VERSION

import numpy as np
import cv2
import json
import time
import queue
import threading
from typing import Dict, Optional, List
from collections import Counter, deque
import logging
import torch

logger = logging.getLogger(__name__)


class FastCCCDRecognition:
    """🔥 CCCD matching với MobileFaceNetV2"""
    
    def __init__(
        self,
        cccd_db_manager,
        mobilefacenet_model=None,
        top_k: int = 5,
        threshold: float = 0.60
    ):
        self.cccd_db = cccd_db_manager
        self.model = mobilefacenet_model
        self.top_k = top_k
        self.threshold = threshold
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        logger.info(f"✅ FastCCCDRecognition initialized (device={self.device}, threshold={threshold})")
    
    def set_model(self, model):
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        try:
          
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
            face_transposed = np.transpose(face_normalized, (2, 0, 1))
            tensor = torch.from_numpy(face_transposed).unsqueeze(0).float()
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"❌ Preprocess error: {e}")
            return None
    
    def _extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            logger.error("❌ Model not loaded!")
            return None
        
        try:
            tensor = self._preprocess_face(face_img)
            if tensor is None:
                return None
            
            with torch.no_grad():
                try:
                    embedding = self.model(tensor)
                except AttributeError:
                    embedding = self.model.forward(tensor)
            
            embedding_np = embedding.cpu().numpy().flatten()
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm
            
            return embedding_np
        except Exception as e:
            logger.error(f"❌ Extract embedding error: {e}")
            return None
    
    def _process_images(
        self,
        track_id: int,
        images: List[np.ndarray],
        person_id: str
    ) -> Optional[Dict]:
        """🔥 XỬ LÝ 5 ẢNH: Trả về result hoặc None"""
        if self.model is None:
            logger.error("❌ Model not loaded!")
            return None
        
        logger.info(f"🔍 Processing {len(images)} images for {person_id}")
        
        # Extract embeddings
        embeddings = []
        for i, img in enumerate(images):
            try:
                if img is None or img.size == 0:
                    continue
                
                emb = self._extract_embedding(img)
                if emb is not None and len(emb) > 0:
                    embeddings.append(emb)
                    
            except Exception as e:
                logger.error(f"   ❌ Image {i+1} error: {e}")
        
        if len(embeddings) == 0:
            logger.warning(f"⚠️ Không extract được embedding nào cho {person_id}")
            return None
        
        logger.info(f"   ✅ Extracted {len(embeddings)}/{len(images)} embeddings")
        
        # Voting search
        candidates_votes = {}
        
        # Auto-detect namespace
        namespaces_to_try = ["face", "CCCD_FACES", "FACE", "face_features"]
        search_namespace = None
        
        for ns in namespaces_to_try:
            if ns in self.cccd_db.indexes:
                search_namespace = ns
                break
        
        if search_namespace is None:
            logger.error(f"❌ No valid face namespace found!")
            return None
        
        for idx, emb in enumerate(embeddings):
            try:
                cccd_result = self.cccd_db.search_vector_with_voting(
                    namespace=search_namespace,
                    query_vector=emb.tolist()
                )
                
                if cccd_result:
                    cccd_id, score = cccd_result
                    
                    if score >= self.threshold:
                        if cccd_id not in candidates_votes:
                            candidates_votes[cccd_id] = []
                        candidates_votes[cccd_id].append(score)
                        
            except Exception as e:
                logger.error(f"   ❌ Search error: {e}")
        
        if not candidates_votes:
            logger.info(f"⚠️ Không match CCCD nào cho {person_id}")
            return None
        
        # Select best match
        best_match = None
        best_score = 0
        
        for cccd_id, scores in candidates_votes.items():
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            combined = (avg_score * 0.7) + (max_score * 0.3)
            
            if combined > best_score:
                best_score = combined
                best_match = cccd_id
        
        if best_match is None:
            return None
        
        logger.info(f"🎯 [MATCH] {person_id} → {best_match} (Score: {best_score:.4f})")
        
        # Get metadata
        cccd_meta = self.cccd_db.get_metadata(best_match)
        if not cccd_meta:
            return None
        
        return {
            'matched': True,
            'track_id': track_id,
            'person_id': person_id,
            'cccd_id': best_match,
            'cccd_confidence': float(best_score),
            'face_match_score': float(best_score),  # 🔥 Thêm field này để dùng cho validation
            'cccd_metadata': {
                'name': cccd_meta.get('name', 'Unknown'),
                'age': cccd_meta.get('age', 'unknown'),
                'gender': cccd_meta.get('gender', 'unknown'),
                'race': cccd_meta.get('race', 'unknown'),
                'cccd_number': cccd_meta.get('cccd_id', best_match),
                'country': cccd_meta.get('country', 'unknown')
            }
        }


class DualStreamManager:
    """
    🔥 QUẢN LÝ 2-SEND SYSTEM + PERIODIC VOTING
    - SEND-1: Khi CCCD match (3-5s)
    - SEND-2: Khi AI consolidation (10-30s)
    - PERIODIC: Mỗi 5 frames
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.person_state = {}
        self.lock = threading.Lock()
        
        # Voting buffers
        self.voting_buffers = {}
        self.last_periodic_send = {}
        self.fully_locked_people = set()

        logger.info("✅ DualStreamManager initialized")
    
    def remap_person_id(self, old_id: str, new_id: str):
        """
        🔥 DI CƯ DỮ LIỆU từ ID tạm sang ID chính thức
        """
        with self.lock:
            if old_id in self.person_state and old_id != new_id:
                logger.info(f"🔄 [REMAP] {old_id} -> {new_id}")
                
                if new_id not in self.person_state:
                    self.person_state[new_id] = self.person_state[old_id]
                    self.person_state[new_id]['person_id'] = new_id
                else:
                    if not self.person_state[new_id].get('cccd_data'):
                        self.person_state[new_id]['cccd_data'] = self.person_state[old_id].get('cccd_data')
                
                del self.person_state[old_id]
            
            # Remap voting buffer
            if old_id in self.voting_buffers and old_id != new_id:
                if new_id not in self.voting_buffers:
                    self.voting_buffers[new_id] = self.voting_buffers[old_id]
                del self.voting_buffers[old_id]
    
    def get_state(self, person_id: str) -> Optional[Dict]:
        """
        🔥 LẤY STATE CỦA PERSON_ID
        Returns: dict hoặc None
        """
        with self.lock:
            return self.person_state.get(person_id)

    def cleanup_old_states(self, active_person_ids: set, max_age_seconds: int = 300):
        """
        🔥 XÓA STATE CŨ (>5 phút không hoạt động)
        """
        with self.lock:
            current_time = time.time()
            to_remove = []
            
            for person_id, state in self.person_state.items():
                # Không xóa nếu vẫn active
                if person_id in active_person_ids:
                    continue
                
                # Xóa nếu quá cũ
                age = current_time - state.get('timestamp', current_time)
                if age > max_age_seconds:
                    to_remove.append(person_id)
            
            for person_id in to_remove:
                del self.person_state[person_id]
                
                # Xóa voting buffer
                if person_id in self.voting_buffers:
                    del self.voting_buffers[person_id]
                
                if person_id in self.last_periodic_send:
                    del self.last_periodic_send[person_id]
            
            if to_remove:
                logger.info(f"🧹 [CLEANUP] Removed {len(to_remove)} old states")


        
    def mark_fully_locked(self, person_id: str):
        """Đánh dấu person đã fully locked - dừng tất cả processing"""
        with self.lock:
            self.fully_locked_people.add(person_id)
            logger.info(f"🔒 [FULLY LOCKED] {person_id} - Stop all AI/CCCD processing")

    def is_fully_locked(self, person_id: str) -> bool:
        """Check có fully locked không"""
        with self.lock:
            return person_id in self.fully_locked_people

    def cleanup_locked_states(self, active_person_ids: set):
        """Xóa locked states khi người biến mất"""
        with self.lock:
            # Xóa những người không còn trong active tracking
            fully_locked_copy = self.fully_locked_people.copy()
            
            for person_id in fully_locked_copy:
                if person_id not in active_person_ids:
                    self.fully_locked_people.discard(person_id)
                    logger.info(f"🗑️ [CLEANUP LOCKED] {person_id}")

    def add_to_voting_buffer(self, person_id: str, attributes: dict):
        """Thêm attributes vào buffer"""
        with self.lock:
            if person_id not in self.voting_buffers:
                self.voting_buffers[person_id] = {
                    'gender': deque(maxlen=5),
                    'age': deque(maxlen=5),
                    'race': deque(maxlen=5),
                    'emotion': deque(maxlen=5),
                    'upper_type': deque(maxlen=5),
                    'lower_type': deque(maxlen=5),
                    'upper_color': deque(maxlen=5),
                    'lower_color': deque(maxlen=5)
                }
            
            buffer = self.voting_buffers[person_id]
            
            buffer['gender'].append(attributes.get('gender', 'unknown'))
            buffer['age'].append(attributes.get('age', 'unknown'))
            buffer['race'].append(attributes.get('race', 'unknown'))
            buffer['emotion'].append(attributes.get('emotion', 'N/A'))
            buffer['upper_type'].append(attributes.get('upper_type', 'Chua xac dinh'))
            buffer['lower_type'].append(attributes.get('lower_type', 'Chua xac dinh'))
            
            # Colors
            upper_color = attributes.get('upper_color')
            lower_color = attributes.get('lower_color')
            
            if upper_color and isinstance(upper_color, (list, tuple)):
                buffer['upper_color'].append(tuple(upper_color))
            else:
                buffer['upper_color'].append(None)
            
            if lower_color and isinstance(lower_color, (list, tuple)):
                buffer['lower_color'].append(tuple(lower_color))
            else:
                buffer['lower_color'].append(None)
    
    def get_voted_attributes(self, person_id: str) -> dict:
        """Vote kết quả từ buffer"""
        with self.lock:
            if person_id not in self.voting_buffers:
                return {
                    'gender': 'unknown',
                    'age': 'unknown',
                    'race': 'unknown',
                    'emotion': 'N/A',
                    'upper_type': 'Chua xac dinh',
                    'lower_type': 'Chua xac dinh',
                    'upper_color': None,
                    'lower_color': None
                }
            
            buffer = self.voting_buffers[person_id]
            voted = {}
            
            # Vote text attributes
            for key in ['gender', 'age', 'race', 'emotion', 'upper_type', 'lower_type']:
                values = buffer[key]
                if len(values) == 0:
                    voted[key] = 'unknown' if key in ['gender', 'age', 'race'] else 'N/A'
                else:
                    valid_values = [v for v in values if v and v not in ['unknown', 'N/A', 'Chua xac dinh']]
                    
                    if valid_values:
                        counter = Counter(valid_values)
                        voted[key] = counter.most_common(1)[0][0]
                    else:
                        voted[key] = 'unknown' if key in ['gender', 'age', 'race'] else 'N/A'
            
            # Vote colors
            for color_key in ['upper_color', 'lower_color']:
                colors = [c for c in buffer[color_key] if c is not None]
                
                if colors:
                    avg_color = tuple(int(np.mean([c[i] for c in colors])) for i in range(3))
                    voted[color_key] = list(avg_color)
                else:
                    voted[color_key] = None
            
            return voted
    
    def should_send_periodic(self, person_id: str, current_frame: int, interval: int = 5) -> bool:
        """Check throttle"""
        with self.lock:
            if person_id not in self.last_periodic_send:
                self.last_periodic_send[person_id] = 0
            
            last_frame = self.last_periodic_send[person_id]
            
            if current_frame - last_frame >= interval:
                self.last_periodic_send[person_id] = current_frame
                return True
            
            return False
    
    def send_periodic_update(
        self, 
        person_id: str, 
        current_frame: int, 
        llm_sender,
        obj_data: dict,
        db_manager
    ) -> bool:
        """🔥 GỬI PERIODIC UPDATE"""
        if not self.should_send_periodic(person_id, current_frame, interval=5):
            return False
        
        final_attrs = obj_data.get('final_attributes')
        if not final_attrs:
            return False
        
        # Validate có data
        has_valid_data = any([
            final_attrs.get('gender') not in [None, 'unknown', ''],
            final_attrs.get('age') not in [None, 'unknown', ''],
            final_attrs.get('race') not in [None, 'unknown', '']
        ])
        
        if not has_valid_data:
            return False
        
        status = obj_data.get('status', 'pending')


        # Check locked status
        db_meta = db_manager.get_metadata(person_id)
        is_locked = (
            db_meta is not None and
            db_meta.get('status') == 'confirmed' and
            db_meta.get('age') and
            db_meta.get('gender') and
            db_meta.get('race')
        )
        
        # Get attributes
        if is_locked:
            attributes = {
                'gender': db_meta.get('gender', 'unknown'),
                'age': db_meta.get('age_onnx', db_meta.get('age', 'unknown')),
                'race': db_meta.get('race', 'unknown'),
                'emotion': final_attrs.get('emotion', 'N/A'),
                'upper_type': final_attrs.get('upper_type', 'Chua xac dinh'),
                'lower_type': final_attrs.get('lower_type', 'Chua xac dinh'),
                'upper_color': final_attrs.get('upper_color'),
                'lower_color': final_attrs.get('lower_color')
            }
            source = "database"
            status = "confirmed"
        else:
            attributes = {
                'gender': final_attrs.get('gender', 'unknown'),
                'age': final_attrs.get('age_onnx', final_attrs.get('age', 'unknown')),
                'race': final_attrs.get('race', 'unknown'),
                'emotion': final_attrs.get('emotion', 'N/A'),
                'upper_type': final_attrs.get('upper_type', 'Chua xac dinh'),
                'lower_type': final_attrs.get('lower_type', 'Chua xac dinh'),
                'upper_color': final_attrs.get('upper_color'),
                'lower_color': final_attrs.get('lower_color')
            }
            source = "live"
        
        # Get CCCD if exists
        cccd_payload = None
        with self.lock:
            if person_id in self.person_state:
                state = self.person_state[person_id]
                if state.get('cccd_data'):
                    cccd = state['cccd_data']['cccd_metadata']
                    cccd_payload = {
                        "id": cccd.get('cccd_number', 'unknown'),
                        "name": cccd.get('name', 'Unknown'),
                        "age": str(cccd.get('age', 'unknown')),
                        "gender": cccd.get('gender', 'unknown'),
                        "country": cccd.get('country', 'unknown')
                    }
        
        # Format colors
        upper_color_str = "N/A"
        lower_color_str = "N/A"
        
        if attributes['upper_color']:
            rgb = attributes['upper_color']
            upper_color_str = f"RGB({rgb[0]},{rgb[1]},{rgb[2]})"
        
        if attributes['lower_color']:
            rgb = attributes['lower_color']
            lower_color_str = f"RGB({rgb[0]},{rgb[1]},{rgb[2]})"
        
        payload = {
            "person_id": person_id,
            "frame": current_frame,
            "status": status,
            "source": source,
            "AI": {
                "gender_ai": attributes['gender'],
                "age_ai": attributes['age'],
                "race_ai": attributes['race'],
                "emotion": attributes['emotion'],
                "clothing": {
                    "upper_type": attributes['upper_type'],
                    "upper_color": upper_color_str,
                    "lower_type": attributes['lower_type'],
                    "lower_color": lower_color_str
                }
            },
            "CCCD": cccd_payload
        }
        
        success = llm_sender.send_json(payload, stream="periodic_update", priority="low")
        
        if success:
            logger.info(f"✅ [PERIODIC] {person_id} at frame {current_frame}")
            return True
        
        return False
    
    def on_cccd_result(self, cccd_result: Dict):
        """Nhận CCCD result"""
        person_id = cccd_result['person_id']
        
        with self.lock:
            if person_id not in self.person_state:
                self.person_state[person_id] = {
                    'person_id': person_id,
                    'track_id': cccd_result.get('track_id'),
                    'send_1_done': False,
                    'send_2_done': False,
                    'cccd_data': None,
                    'ai_data': None,
                    'timestamp': time.time()
                }
            
            self.person_state[person_id]['cccd_data'] = cccd_result
            self.person_state[person_id]['cccd_name'] = cccd_result['cccd_metadata'].get('name', 'Unknown')

    def send_first(self, person_id: str, llm_sender, obj_data: dict = None) -> bool: 
        """SEND-1: CCCD match"""
        with self.lock:
            if person_id not in self.person_state:
                return False
            
            state = self.person_state[person_id]
            
            if state['send_1_done']:
                return False
            
            if state['cccd_data'] is None:
                return False
            
            cccd = state['cccd_data']['cccd_metadata']
            status = "pending"  # default
            if obj_data:
                status = obj_data.get('status', 'pending')

            payload = {
                "person_id": person_id,
                "status": status,
                "AI": {
                    "age_ai": "unknown",
                    "gender_ai": "unknown",
                    "race_ai": "unknown"
                },
                "CCCD": {
                    "id": cccd.get('cccd_number', 'unknown'),
                    "name": cccd.get('name', 'Unknown'),
                    "age": str(cccd.get('age', 'unknown')),
                    "gender": cccd.get('gender', 'unknown'),
                    "country": cccd.get('country', 'unknown')
                }
            }
            
            success = llm_sender.send_json(payload, stream="send_1_cccd_match", priority="high")
            
            if success:
                state['send_1_done'] = True
                logger.info(f"📤 [SEND-1] ✅ {person_id}")
                return True
            
            return False
    
    def send_first_no_match(self, person_id: str, llm_sender,obj_data: dict = None) -> bool:
        """SEND-1: No CCCD match"""
        with self.lock:
            if person_id not in self.person_state:
                self.person_state[person_id] = {
                    'person_id': person_id,
                    'track_id': None,
                    'send_1_done': False,
                    'send_2_done': False,
                    'cccd_data': None,
                    'ai_data': None,
                    'timestamp': time.time()
                }
            
            state = self.person_state[person_id]
            
            if state['send_1_done']:
                return False
            status = "pending"  # default
            if obj_data:
                status = obj_data.get('status', 'pending')

            payload = {
                "person_id": person_id,
                "status": status,
                "AI": {
                    "age_ai": "unknown",
                    "gender_ai": "unknown",
                    "race_ai": "unknown"
                },
                "CCCD": None
            }
            
            success = llm_sender.send_json(payload, stream="send_1_cccd_unknown", priority="normal")
            
            if success:
                state['send_1_done'] = True
                logger.info(f"📤 [SEND-1] ✅ {person_id} (no CCCD)")
                return True
            
            return False
    
    def send_second_with_ai(self, person_id: str, ai_attributes: Dict, llm_sender) -> bool:
        """SEND-2: AI + CCCD"""
        with self.lock:
            if person_id not in self.person_state:
                return False
            
            state = self.person_state[person_id]
            
            if state['send_2_done']:
                return False
            
            if not state['send_1_done']:
                return False
            
            state['ai_data'] = ai_attributes

            status = "pending"  # default
            if obj_data:
                status = obj_data.get('status', 'pending')

            payload = {
                "person_id": person_id,
                "status": status,
                "AI": {
                    "age_ai": ai_attributes.get('age', 'unknown'),
                    "gender_ai": ai_attributes.get('gender', 'unknown'),
                    "race_ai": ai_attributes.get('race', 'unknown')
                },
                "CCCD": None
            }
            
            if state['cccd_data']:
                cccd = state['cccd_data']['cccd_metadata']
                payload['CCCD'] = {
                    "id": cccd.get('cccd_number', 'unknown'),
                    "name": cccd.get('name', 'Unknown'),
                    "age": str(cccd.get('age', 'unknown')),
                    "gender": cccd.get('gender', 'unknown'),
                    "country": cccd.get('country', 'unknown')
                }
            
            success = llm_sender.send_json(payload, stream="send_2_final_update", priority="normal")
            
            if success:
                state['send_2_done'] = True
                logger.info(f"📤 [SEND-2] ✅ {person_id}")
                return True
            
            return False


class LLMSender:
    """Gửi JSON cho LLM server"""
    
    def __init__(self, endpoint=None, timeout=5, max_retries=1):
        self.endpoint = endpoint or "http://localhost:8000/receive-jsonl"
        self.timeout = timeout
        self.max_retries = max_retries
        self.total_sent = 0
        self.total_failed = 0
        self.stats_by_stream = {}
        logger.info(f"📡 LLM Endpoint: {self.endpoint}")
    
    def send_json(self, payload, stream="tracking", priority="normal", retry_count=0):
        """Gửi JSON payload"""
        try:
            import requests
        except ImportError:
            logger.warning("⚠️ Thiếu requests")
            return False
        
        full_payload = {
            **payload,
            "metadata": {
                "stream": stream,
                "priority": priority,
                "timestamp": time.time()
            }
        }
        
        try:
            response = requests.post(
                self.endpoint,
                json=full_payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                self.total_sent += 1
                
                if stream not in self.stats_by_stream:
                    self.stats_by_stream[stream] = {'success': 0, 'failed': 0}
                
                self.stats_by_stream[stream]['success'] += 1
                
                logger.info(f"✅ [LLM] [{stream}] {payload.get('person_id')}")
                return True
            else:
                self.total_failed += 1
                logger.warning(f"⚠️ [LLM] HTTP {response.status_code}")
                return False
        
        except Exception as e:
            self.total_failed += 1
            logger.error(f"❌ [LLM] {type(e).__name__}")
            return False