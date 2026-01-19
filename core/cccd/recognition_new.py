# core/cccd/recognition_new.py - COMPLETE FIXED VERSION
#Gửi LLM JSON với 3 giai đoạn
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
        threshold: float = 0.55
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
    
    def _process_images_improved(self, track_id: int, images: List[np.ndarray], person_id: str) -> Optional[Dict]:
        """
        CCCD matching: Gửi hết 200 frame, per-batch confirmation (2 matches)
        """
        if self.model is None:
            logger.error("No model loaded!")
            return None
        
        logger.info(f"Processing {len(images)} images for {person_id}")
        
        embeddings_with_quality = []
        
        for i, img in enumerate(images):
            try:
                if img is None or img.size == 0:
                    continue
                
                emb = self._extract_embedding(img)
                if emb is None or len(emb) == 0:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = np.mean(gray)
                contrast = np.std(gray)
                
                quality = (
                    min(blur / 150, 1.0) * 0.5 +
                    (1 - abs(brightness - 127) / 127) * 0.3 +
                    min(contrast / 50, 1.0) * 0.2
                )
                
                embeddings_with_quality.append((emb, quality))
                
            except Exception as e:
                logger.error(f"   Image {i+1} error: {e}")
        
        if len(embeddings_with_quality) == 0:
            logger.warning(f"No embeddings extracted for {person_id}")
            return None
        
        unique_embeddings = self._deduplicate_embeddings(embeddings_with_quality, similarity_threshold=0.95)
        
        logger.info(f"   Extracted {len(embeddings_with_quality)} -> Deduplicated {len(unique_embeddings)}")
        
        unique_embeddings.sort(key=lambda x: x[1], reverse=True)
        top_embeddings = unique_embeddings[:5]
        
        namespaces_to_try = ["face", "CCCD_FACES", "FACE", "face_features"]
        search_namespace = None
        
        for ns in namespaces_to_try:
            if ns in self.cccd_db.indexes:
                search_namespace = ns
                break
        
        if search_namespace is None:
            logger.error("No valid face namespace found!")
            return None
        
        match_votes = {}
        CONFIRMATION_THRESHOLD = 0.58  # 🔥 Hạ từ 0.6 → 0.58 để match CCCD embedding score
        best_match = None
        best_score = None
        
        for idx, (emb, quality) in enumerate(top_embeddings):
            try:
                cccd_result = self.cccd_db.search_vector_with_voting(
                    namespace=search_namespace,
                    query_vector=emb.tolist()
                )
                
                if cccd_result:
                    cccd_id, score = cccd_result
                    
                    if score >= CONFIRMATION_THRESHOLD:
                        best_match = cccd_id
                        best_score = score
                        logger.info(f"CONFIRMED: {person_id} -> {best_match} (score={best_score:.4f})")
                        break
                    elif score >= self.threshold:
                        if cccd_id not in match_votes:
                            match_votes[cccd_id] = []
                        match_votes[cccd_id].append(score)
                        
            except Exception as e:
                logger.error(f"   Search error: {e}")
        else:
            if not match_votes:
                logger.info(f"No CCCD match for {person_id}")
                return None
            
            best_match = max(match_votes.items(), key=lambda x: np.mean(x[1]))[0]
            best_score = np.mean(match_votes[best_match])
            logger.info(f"PARTIAL MATCH: {person_id} -> {best_match} (votes={len(match_votes[best_match])}, score={best_score:.4f})")
            
            if best_score < self.threshold:
                logger.info(f"Score {best_score:.4f} below threshold {self.threshold}")
                return None
        
        # Safety check - best_match should always be defined by now
        if best_match is None:
            logger.warning(f"⚠️ No valid CCCD match found for {person_id}")
            return None
        
        # Get metadata
        cccd_meta = self.cccd_db.get_metadata(best_match)
        if not cccd_meta:
            logger.error(f"❌ [METADATA] No metadata found for CCCD {best_match}")
            return None
        
        # 🔥 VALIDATION: Check metadata có đầy đủ thông tin không
        name = cccd_meta.get('name', 'Unknown')
        if not name or name == 'Unknown' or str(name).strip() == '':
            logger.warning(f"⚠️ [METADATA] CCCD {best_match} missing or invalid name: '{name}'")
            # Vẫn return nhưng log cảnh báo
        
        logger.info(f"✅ [CCCD DETAILS] {best_match} - Name: {name}, Age: {cccd_meta.get('age')}, Gender: {cccd_meta.get('gender')}")
        
        return {
            'matched': True,
            'track_id': track_id,
            'person_id': person_id,
            'cccd_id': best_match,
            'cccd_confidence': float(best_score),
            'cccd_metadata': {
                'name': name,
                'age': cccd_meta.get('age', 'unknown'),
                'gender': cccd_meta.get('gender', 'unknown'),
                'race': cccd_meta.get('race', 'unknown'),
                'cccd_number': cccd_meta.get('cccd_id', best_match),
                'country': cccd_meta.get('country', 'unknown')
            }
        }
    
    def _deduplicate_embeddings(self, embeddings_with_quality, similarity_threshold=0.98):
        """
        🔥 ADAPTIVE DEDUPLICATION - Thông minh hơn!
        
        Logic:
        - Threshold CAO (0.98) = Chỉ loại bỏ ảnh GẦN GIỐNG HỆT NHAU
        - Giữ lại ảnh tương tự nhưng KHÔNG GIỐNG HỆT (0.95-0.98)
        - Bảo vệ trường hợp người đứng yên nhưng lighting thay đổi
        
        Ví dụ:
        - Similarity 0.99 → LOẠI BỎ (ảnh duplicate thật sự)
        - Similarity 0.96 → GIỮ LẠI (góc/ánh sáng khác nhau)
        """
        if len(embeddings_with_quality) <= 1:
            return embeddings_with_quality
        
        unique = []
        
        # 🔥 PHÂN TÍCH ĐỘ ĐA DẠNG TỔNG THỂ
        all_similarities = []
        for i in range(len(embeddings_with_quality)):
            for j in range(i + 1, len(embeddings_with_quality)):
                emb1, _ = embeddings_with_quality[i]
                emb2, _ = embeddings_with_quality[j]
                
                sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
                )
                all_similarities.append(sim)
        
        # 🔥 ADAPTIVE THRESHOLD dựa vào độ đa dạng
        if all_similarities:
            avg_sim = np.mean(all_similarities)
            
            if avg_sim > 0.95:
                # Trường hợp: Người đứng yên → Dùng threshold CAO để giữ lại nhiều hơn
                adaptive_threshold = 0.99
                logger.debug(f"   📊 [ADAPTIVE] High similarity detected (avg={avg_sim:.3f}) → Using threshold={adaptive_threshold}")
            else:
                # Trường hợp: Người di chuyển → Dùng threshold THẤP HƠN
                adaptive_threshold = 0.97
                logger.debug(f"   📊 [ADAPTIVE] Good diversity (avg={avg_sim:.3f}) → Using threshold={adaptive_threshold}")
        else:
            adaptive_threshold = similarity_threshold
        
        # 🔥 DEDUPLICATION với ADAPTIVE THRESHOLD
        for emb, quality in embeddings_with_quality:
            is_unique = True
            
            for existing_emb, _ in unique:
                similarity = np.dot(emb, existing_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(existing_emb) + 1e-8
                )
                
                # 🔥 CHỈ LOẠI BỎ NẾU GIỐNG GẦN NHƯ HỆT NHAU
                if similarity > adaptive_threshold:
                    is_unique = False
                    logger.debug(f"      ❌ Removed: similarity={similarity:.4f} > {adaptive_threshold:.4f}")
                    break
            
            if is_unique:
                unique.append((emb, quality))
        
        return unique
        
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
class DualStreamManager:
    """
    🔥 3-STAGE SEND SYSTEM
    - SEND-1: Gender/Age/Race (when confidence >= 0.8)
    - SEND-2: CCCD match (with current AI data)
    - SEND-3: Final confirmed (full data)
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.person_state = {}
        self.lock = threading.Lock()
        
        # Voting buffers
        self.voting_buffers = {}
        self.last_periodic_send = {}
        self.fully_locked_people = set()

        logger.info("✅ DualStreamManager initialized (3-Stage)")
    
    def remap_person_id(self, old_id: str, new_id: str):
        """DI CƯ DỮ LIỆU từ ID tạm sang ID chính thức"""
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
            
            if old_id in self.voting_buffers and old_id != new_id:
                if new_id not in self.voting_buffers:
                    self.voting_buffers[new_id] = self.voting_buffers[old_id]
                del self.voting_buffers[old_id]
    
    def get_state(self, person_id: str) -> Optional[Dict]:
        """LẤY STATE CỦA PERSON_ID"""
        with self.lock:
            return self.person_state.get(person_id)

    def cleanup_old_states(self, active_person_ids: set, max_age_seconds: int = 300):
        """XÓA STATE CŨ (>5 phút không hoạt động)"""
        with self.lock:
            current_time = time.time()
            to_remove = []
            
            for person_id, state in self.person_state.items():
                if person_id in active_person_ids:
                    continue
                
                age = current_time - state.get('timestamp', current_time)
                if age > max_age_seconds:
                    to_remove.append(person_id)
            
            for person_id in to_remove:
                del self.person_state[person_id]
                
                if person_id in self.voting_buffers:
                    del self.voting_buffers[person_id]
                
                if person_id in self.last_periodic_send:
                    del self.last_periodic_send[person_id]
            
            if to_remove:
                logger.info(f"🧹 [CLEANUP] Removed {len(to_remove)} old states")
        
    def mark_fully_locked(self, person_id: str):
        """Đánh dấu person đã fully locked"""
        with self.lock:
            self.fully_locked_people.add(person_id)
            logger.info(f"🔒 [FULLY LOCKED] {person_id}")

    def is_fully_locked(self, person_id: str) -> bool:
        """Check fully locked"""
        with self.lock:
            return person_id in self.fully_locked_people

    def cleanup_locked_states(self, active_person_ids: set):
        """Xóa locked states khi người biến mất"""
        with self.lock:
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
            
            for color_key in ['upper_color', 'lower_color']:
                colors = [c for c in buffer[color_key] if c is not None]
                
                if colors:
                    avg_color = tuple(int(np.mean([c[i] for c in colors])) for i in range(3))
                    voted[color_key] = list(avg_color)
                else:
                    voted[color_key] = None
            
            return voted
    
    # ============================================================
    # PERIODIC UPDATE - DISABLED (Không gửi mỗi 5 frame nữa)
    # Chỉ gửi 3 request: SEND-1 (gender), SEND-2 (CCCD), SEND-3 (confirmed)
    # ============================================================
    
    def should_send_periodic(self, person_id: str, current_frame: int, interval: int = 5) -> bool:
        """DISABLED - Không sử dụng periodic update nữa"""
        return False
    
    def send_periodic_update(
        self, 
        person_id: str, 
        current_frame: int, 
        llm_sender,
        obj_data: dict,
        db_manager
    ) -> bool:
        """
        DISABLED - Không gửi periodic update nữa
        Chỉ gửi 3 request chính thức: SEND-1, SEND-2, SEND-3
        """
        return False
    
    def on_cccd_result(self, cccd_result: Dict):
        """Nhận CCCD result - Lưu metadata đầy đủ vào DB"""
        person_id = cccd_result['person_id']
        
        with self.lock:
            if person_id not in self.person_state:
                self.person_state[person_id] = {
                    'person_id': person_id,
                    'track_id': cccd_result.get('track_id'),
                    'send_1_done': False,  # Gender/Age/Race sent
                    'send_2_done': False,  # CCCD sent
                    'send_3_done': False,  # Final confirmed sent
                    'total_llm_requests': 0,  # Counter for total LLM requests
                    'cccd_data': None,
                    'ai_data': None,
                    'timestamp': time.time()
                }
            
            self.person_state[person_id]['cccd_data'] = cccd_result
            cccd_meta = cccd_result['cccd_metadata']
            self.person_state[person_id]['cccd_name'] = cccd_meta.get('name', 'Unknown')
        
        # 🔥 BƯỚC 1: KIỂM TRA ĐÃ CÓ CCCD TỪ TRƯỚC KHÔNG (tránh lưu lộn)
        existing_metadata = self.db_manager.get_metadata(person_id)
        already_has_cccd = existing_metadata and existing_metadata.get('cccd_matched', False)
        
        # Lưu CCCD metadata đầy đủ vào DB
        if not person_id.startswith('Temp_'):
            try:
                # 🔥 LUÔN LƯU CCCD (không check face_match_score)
                # Chỉ kiểm tra để tránh LƯU LỘN người khác
                if not already_has_cccd:
                    cccd_info = {
                        'name': cccd_meta.get('name', 'Unknown'),
                        'age': cccd_meta.get('age', 'Unknown'),
                        'gender': cccd_meta.get('gender', 'Unknown'),
                        'race': cccd_meta.get('race', 'Unknown'),
                        'country': cccd_meta.get('country', 'Unknown'),
                        'cccd_number': cccd_result.get('cccd_id', 'Unknown'),
                        'confidence': cccd_result.get('cccd_confidence', 0.0)
                    }
                    
                    self.db_manager.save_cccd_metadata(person_id, cccd_info)
                    self.db_manager.save_all_databases()
                    logger.info(
                        f"✅ [DB SAVE CCCD] {person_id} - Name: {cccd_meta.get('name')}, "
                        f"Age: {cccd_meta.get('age')}, Gender: {cccd_meta.get('gender')}"
                    )
                else:
                    logger.warning(
                        f"⚠️ [SKIP CCCD SAVE] {person_id}: Already has CCCD metadata "
                        f"({existing_metadata.get('cccd_name', 'Unknown')}). "
                        f"Preventing duplicate save."
                    )
            
            except Exception as e:
                logger.error(f"[DB SAVE CCCD ERROR] {person_id}: {e}")

    def send_stage1_ai_attributes(self, person_id: str, ai_attributes: Dict, llm_sender, obj_data: dict = None) -> bool:
        """
        🔥 SEND-1: GỬI GENDER/AGE/RACE (khi confidence >= 0.8)
        Gửi ngay khi có AI attributes với độ tin cậy cao
        """
        with self.lock:
            if person_id not in self.person_state:
                self.person_state[person_id] = {
                    'person_id': person_id,
                    'track_id': None,
                    'send_1_done': False,
                    'send_2_done': False,
                    'send_3_done': False,
                    'total_llm_requests': 0,  # Counter for total LLM requests
                    'cccd_data': None,
                    'ai_data': None,
                    'timestamp': time.time()
                }
            
            state = self.person_state[person_id]
            
            # Kiểm tra đã gửi đủ 3 requests chưa
            if state['total_llm_requests'] >= 3:
                return False
            
            # Đã gửi rồi thì skip
            if state['send_1_done']:
                return False
            
            # Kiểm tra confidence
            gender = ai_attributes.get('gender', 'unknown')
            age = ai_attributes.get('age', 'unknown')
            race = ai_attributes.get('race', 'unknown')
            
            # Lấy confidence scores
            gender_conf = ai_attributes.get('gender_confidence', 0)
            age_conf = ai_attributes.get('age_confidence', 0)
            race_conf = ai_attributes.get('race_confidence', 0)
            
            # Kiểm tra có ít nhất 1 attribute với confidence >= 0.7 (gender) hoặc >= 0.8 (age/race)
            has_high_conf = (
                (gender != 'unknown' and gender_conf >= 0.7) or
                (age != 'unknown' and age_conf >= 0.8) or
                (race != 'unknown' and race_conf >= 0.8)
            )
            
            if not has_high_conf:
                return False
            
            # Lưu AI data
            state['ai_data'] = ai_attributes
            
            status = "pending"
            if obj_data:
                status = obj_data.get('status', 'pending')
            
            payload = {
                "person_id": person_id,
                "status": status,
                "AI": {
                    "gender_ai": gender,
                    "age_ai": age,
                    "race_ai": race,
                    "confidence": {
                        "gender": float(gender_conf),
                        "age": float(age_conf),
                        "race": float(race_conf)
                    }
                },
                "CCCD": None
            }
            
            success = llm_sender.send_json(payload, stream="send_1_ai_attributes", priority="high")
            
            if success:
                state['send_1_done'] = True
                state['total_llm_requests'] += 1
                logger.info(f"📤 [SEND-1] ✅ {person_id} - AI Attributes (G:{gender_conf:.2f} A:{age_conf:.2f} R:{race_conf:.2f}) [Total: {state['total_llm_requests']}/3]")
                return True
            
            return False

    def send_stage2_cccd_match(self, person_id: str, llm_sender, obj_data: dict = None) -> bool:
        """
        🔥 SEND-2: GỬI CCCD MATCH (kèm AI data hiện tại nếu có)
        Gửi khi match được CCCD
        """
        with self.lock:
            if person_id not in self.person_state:
                return False
            
            state = self.person_state[person_id]
            
            # Kiểm tra đã gửi đủ 3 requests chưa
            if state['total_llm_requests'] >= 3:
                return False
            
            # Đã gửi CCCD rồi thì skip
            if state['send_2_done']:
                return False
            
            # Chưa có CCCD data thì không gửi
            if state['cccd_data'] is None:
                return False
            
            cccd = state['cccd_data']['cccd_metadata']
            
            status = "pending"
            if obj_data:
                status = obj_data.get('status', 'pending')
            
            # Lấy AI data nếu có
            ai_payload = {
                "gender_ai": "unknown",
                "age_ai": "unknown",
                "race_ai": "unknown"
            }
            
            if state['ai_data']:
                ai_payload = {
                    "gender_ai": state['ai_data'].get('gender', 'unknown'),
                    "age_ai": state['ai_data'].get('age', 'unknown'),
                    "race_ai": state['ai_data'].get('race', 'unknown')
                }
            
            payload = {
                "person_id": person_id,
                "status": status,
                "AI": ai_payload,
                "CCCD": {
                    "id": cccd.get('cccd_number', 'unknown'),
                    "name": cccd.get('name', 'Unknown'),
                    "age": str(cccd.get('age', 'unknown')),
                    "gender": cccd.get('gender', 'unknown'),
                    "country": cccd.get('country', 'unknown')
                }
            }
            
            success = llm_sender.send_json(payload, stream="send_2_cccd_match", priority="high")
            
            if success:
                state['send_2_done'] = True
                state['total_llm_requests'] += 1
                logger.info(f"📤 [SEND-2] ✅ {person_id} - CCCD: {cccd.get('name')} [Total: {state['total_llm_requests']}/3]")
                return True
            
            return False

    def send_stage3_confirmed(self, person_id: str, ai_attributes: Dict, llm_sender, obj_data: dict = None) -> bool:
        """
        🔥 SEND-3: GỬI FULL DATA KHI CONFIRMED
        Gửi khi person đã có status 'confirmed' và đầy đủ thông tin
        """
        with self.lock:
            if person_id not in self.person_state:
                return False
            
            state = self.person_state[person_id]
            
            # Kiểm tra đã gửi đủ 3 requests chưa
            if state['total_llm_requests'] >= 3:
                return False
            
            # Đã gửi confirmed rồi thì skip
            if state['send_3_done']:
                return False
            
            # Kiểm tra điều kiện confirmed
            if not obj_data or obj_data.get('status') != 'confirmed':
                return False
            
            # Kiểm tra có đầy đủ AI data
            gender = ai_attributes.get('gender', 'unknown')
            age = ai_attributes.get('age', 'unknown')
            race = ai_attributes.get('race', 'unknown')
            
            if gender == 'unknown' or age == 'unknown' or race == 'unknown':
                return False
            
            # Update AI data
            state['ai_data'] = ai_attributes
            
            payload = {
                "person_id": person_id,
                "status": "confirmed",
                "AI": {
                    "gender_ai": gender,
                    "age_ai": age,
                    "race_ai": race,
                    "confidence": {
                        "gender": float(ai_attributes.get('gender_confidence', 0)),
                        "age": float(ai_attributes.get('age_confidence', 0)),
                        "race": float(ai_attributes.get('race_confidence', 0))
                    }
                },
                "CCCD": None
            }
            
            # Thêm CCCD nếu có
            if state['cccd_data']:
                cccd = state['cccd_data']['cccd_metadata']
                payload['CCCD'] = {
                    "id": cccd.get('cccd_number', 'unknown'),
                    "name": cccd.get('name', 'Unknown'),
                    "age": str(cccd.get('age', 'unknown')),
                    "gender": cccd.get('gender', 'unknown'),
                    "country": cccd.get('country', 'unknown')
                }
            
            success = llm_sender.send_json(payload, stream="send_3_final_confirmed", priority="normal")
            
            if success:
                state['send_3_done'] = True
                state['total_llm_requests'] += 1
                logger.info(f"📤 [SEND-3] ✅ {person_id} - CONFIRMED FULL DATA [Total: {state['total_llm_requests']}/3]")
                return True
            
            return False
    
    # ============================================================
    # DEPRECATED - Giữ lại để backward compatibility
    # ============================================================
    def send_first(self, person_id: str, llm_sender, obj_data: dict = None) -> bool:
        """⚠️ DEPRECATED - Use send_stage2_cccd_match instead"""
        return self.send_stage2_cccd_match(person_id, llm_sender, obj_data)
    
    def send_first_no_match(self, person_id: str, llm_sender, obj_data: dict = None) -> bool:
        """⚠️ DEPRECATED - Không cần gửi no-match nữa"""
        return False
    
    def send_second_with_ai(self, person_id: str, ai_attributes: Dict, llm_sender) -> bool:
        """⚠️ DEPRECATED - Use send_stage3_confirmed instead"""
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
            # logger.error(f"❌ [LLM] {type(e).__name__}")
            return False
    
    def send_long_absence_notification(self, person_id: str, duration_minutes: float, metadata: dict, llm_sender) -> bool:
        """
        Gui thong bao khi nguoi di lau (> 30 phut) roi quay lai.
        
        Args:
            person_id: ID nguoi
            duration_minutes: Thoi gian vang mat (phut)
            metadata: Du lieu AI (gender, age, race, timestamps)
            llm_sender: LLM sender instance
        """
        logger.info(f"🤖 [LONG ABSENCE] {person_id} - Duration: {duration_minutes:.1f} min")
        
        # Chuẩn bị payload
        payload = {
            'event_type': 'long_absence_return',
            'person_id': person_id,
            'duration_minutes': round(duration_minutes, 1),
            'first_seen': metadata.get('first_seen'),
            'last_seen': metadata.get('last_seen'),
            'attributes': metadata.get('attributes', {}),
            'context': metadata.get('context', 'long_absence'),
            'reason': metadata.get('reason', f'Returned after {duration_minutes:.0f} minutes'),
            'timestamp': time.time()
        }
        
        # Gui qua LLM sender
        try:
            if hasattr(llm_sender, 'send_custom_message'):
                # Format message cho LLM
                attrs = payload['attributes']
                message = (
                    f"⚠️ Long Absence Alert\n"
                    f"Person {person_id} has returned after {duration_minutes:.1f} minutes.\n\n"
                    f"Profile:\n"
                    f"  - Gender: {attrs.get('gender', 'unknown')}\n"
                    f"  - Age: {attrs.get('age', 'unknown')}\n"
                    f"  - Race: {attrs.get('race', 'unknown')}\n\n"
                    f"Timeline:\n"
                    f"  - First seen: {payload['first_seen']}\n"
                    f"  - Last seen: {payload['last_seen']}\n\n"
                    f"Please review and provide appropriate response."
                )
                
                success = llm_sender.send_custom_message(message)
                
                if success:
                    logger.info(f"✅ [LONG ABSENCE] Sent to LLM successfully")
                    return True
                else:
                    logger.warning(f"⚠️ [LONG ABSENCE] Failed to send to LLM")
                    return False
            else:
                logger.warning(f"⚠️ [LONG ABSENCE] LLM sender missing send_custom_message method")
                return False
                
        except Exception as e:
            logger.error(f"❌ [LONG ABSENCE] Error: {e}")
            return False