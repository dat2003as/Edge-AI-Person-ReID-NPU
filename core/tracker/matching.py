# core/tracker/matching.py
import time
import logging
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import config
from .utils import TrackerUtils

logger = logging.getLogger(__name__)


class ConfirmedPersonMatcher:
    """
    Quan ly re-matching cho confirmed persons.
    Dung spatial + temporal + face vector de re-match neu nguoi di xa roi tro lai.
    
    Immutable attributes: gender, age, race, name (chi luu 1 lan khi confirm)
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def vote_person_name(self, obj_data: dict) -> str:
        """
        Vote tên từ history attributes.
        Priority:
        1. cccd_name (từ CCCD recognition)
        2. Tên xuất hiện nhiều nhất trong history
        3. 'Unknown' nếu không có
        
        Return: Tên được vote
        """
        from collections import Counter
        
        names = []
        
        # 1. Ưu tiên CCCD name
        history_attrs = obj_data.get('history_attributes', [])
        for attr in history_attrs:
            cccd_name = attr.get('cccd_name')
            if cccd_name and cccd_name != 'Unknown':
                names.append(cccd_name)
        
        # 2. Nếu có tên từ CCCD, dùng ngay
        if names:
            name_counts = Counter(names)
            voted_name = name_counts.most_common(1)[0][0]
            logger.debug(f"📛 [VOTE NAME] From CCCD: {voted_name} (appearances: {name_counts[voted_name]})")
            return voted_name
        
        # 3. Fallback: Unknown
        logger.debug(f"📛 [VOTE NAME] No valid name found, using 'Unknown'")
        return 'Unknown'
    
    def save_confirmed_attributes(self, person_id: str, obj_data: dict):
        """
        Luu attributes bat bien khi person duoc confirm.
        - first_seen_time: lan dau tien thay nguoi nay (immutable)
        - last_seen_time: lan cuoi cung thay (mutable)
        - confirmed_name: ten duoc vote tu history (immutable)
        - confirmed_gender/age/race: immutable attributes
        """
        if person_id.startswith("Temp_"):
            return
        
        final_attrs = obj_data.get('final_attributes', {})
        
        # Lay attributes hien tai
        gender = final_attrs.get('gender', 'unknown')
        age = final_attrs.get('age_onnx', final_attrs.get('age', 'unknown'))
        race = final_attrs.get('race', 'unknown')
        gender_conf = final_attrs.get('gender_confidence', 0.0)
        age_conf = final_attrs.get('age_confidence', 0.0)
        race_conf = final_attrs.get('race_confidence', 0.0)
        
        # Vote tên từ history
        voted_name = self.vote_person_name(obj_data)
        
        # Cap nhat DB
        metadata = self.db_manager.get_metadata(person_id)
        
        current_time = datetime.now().isoformat()
        
        # Chi luu neu chua co (immutable)
        if 'confirmed_gender' not in metadata:
            metadata['confirmed_gender'] = gender
            metadata['gender_confidence'] = gender_conf
        
        if 'confirmed_age' not in metadata:
            metadata['confirmed_age'] = age
            metadata['age_confidence'] = age_conf
        
        if 'confirmed_race' not in metadata:
            metadata['confirmed_race'] = race
            metadata['race_confidence'] = race_conf
        
        # Luu confirmed_name (immutable - chi set lan dau)
        if 'confirmed_name' not in metadata:
            metadata['confirmed_name'] = voted_name
            logger.info(f"📛 [CONFIRMED NAME] {person_id} - Set name: {voted_name}")
        else:
            logger.debug(
                f"📛 [CONFIRMED NAME PROTECTED] {person_id} - "
                f"Kept original: {metadata['confirmed_name']} (NOT updated to {voted_name})"
            )
        
        # Luu first_seen_time (immutable - chi set lan dau, KHONG BAO GIO THAY DOI)
        if 'first_seen_time' not in metadata:
            metadata['first_seen_time'] = current_time
            logger.info(f"📅 [FIRST SEEN] {person_id} - Set first_seen: {current_time}")
        else:
            # Bao ve first_seen - dam bao khong bi ghi de
            logger.debug(
                f"📅 [FIRST SEEN PROTECTED] {person_id} - "
                f"Kept original: {metadata['first_seen_time']} (NOT updated to {current_time})"
            )
        
        # Update last_seen_time (mutable - update moi lan)
        old_last_seen = metadata.get('last_seen_time', 'never')
        metadata['last_seen_time'] = current_time
        
        # Tinh duration neu co first_seen
        if 'first_seen_time' in metadata:
            try:
                first_dt = datetime.fromisoformat(metadata['first_seen_time'])
                last_dt = datetime.fromisoformat(current_time)
                duration_minutes = (last_dt - first_dt).total_seconds() / 60.0
                logger.debug(
                    f"⏱️  [DURATION] {person_id} - Total: {duration_minutes:.1f} min "
                    f"(first: {metadata['first_seen_time']}, last: {current_time})"
                )
            except:
                pass
        
        self.db_manager.update_metadata(person_id, metadata)
        
        logger.info(
            f"✅ [SAVE CONFIRMED] {person_id} - {metadata.get('confirmed_name', 'Unknown')}\n"
            f"   Gender: {gender} (conf={gender_conf:.2f})\n"
            f"   Age: {age} (conf={age_conf:.2f})\n"
            f"   Race: {race} (conf={race_conf:.2f})\n"
            f"   First seen: {metadata.get('first_seen_time')}\n"
            f"   Last seen: {metadata['last_seen_time']}"
        )
    
    def check_confirmed_person_rematch(
        self,
        track_id: int,
        bbox: list,
        face_vector: Optional[np.ndarray],
        reid_vector: Optional[np.ndarray],
        current_attributes: dict
    ) -> Optional[Tuple[str, float, str]]:
        """
        Kiem tra xem track ID hien tai co phai la re-match cua confirmed person khong.
        
        Return: (person_id, confidence_score, source) hoac None
        
        CASCADE LOGIC (scene < 3m, khong can spatial):
        1. Temporal: < TEMPORAL_MATCHING_WINDOW seconds
        2. PRIMARY: Face vector similarity >= 0.65 (neu co)
        3. FALLBACK: ReID vector similarity >= 0.55 (neu khong co face)
        4. Attributes: Soft check (chi reject neu confidence cao)
        """
        
        # Lay all confirmed persons tu DB
        all_metadata = self._get_all_confirmed_persons()
        
        if not all_metadata:
            return None
        
        best_match = None
        best_score = 0.0
        current_time = time.time()
        
        for person_id, metadata in all_metadata.items():
            # Skip neu khong co thong tin last_seen
            last_seen_str = metadata.get('last_seen_time')
            
            if not last_seen_str:
                continue
            
            # 1. TEMPORAL CHECK
            try:
                last_seen_time = datetime.fromisoformat(last_seen_str).timestamp()
            except:
                continue
            
            time_gap = current_time - last_seen_time
            if time_gap > config.TEMPORAL_MATCHING_WINDOW:
                # Qua lau roi, skip
                continue
            
            # 2. CASCADE VECTOR MATCHING
            vector_score = 0.0
            vector_source = ""
            
            # PRIMARY: Face vector (priority)
            if face_vector is not None:
                face_match = self.db_manager.search_vector_with_voting(
                    config.FACE_NAMESPACE, face_vector
                )
                if face_match:
                    matched_id, matched_score = face_match
                    if matched_id == person_id and matched_score >= config.CONFIRMED_FACE_SIMILARITY_THRESHOLD:
                        vector_score = matched_score
                        vector_source = "FACE"
            
            # FALLBACK: ReID vector (neu khong co face hoac face score thap)
            if vector_score < config.CONFIRMED_FACE_SIMILARITY_THRESHOLD and reid_vector is not None:
                reid_match = self.db_manager.search_vector_with_voting(
                    config.REID_NAMESPACE, reid_vector
                )
                if reid_match:
                    matched_id, matched_score = reid_match
                    if matched_id == person_id and matched_score >= config.CONFIRMED_REID_SIMILARITY_THRESHOLD:
                        vector_score = matched_score
                        vector_source = "REID"
            
            if vector_score == 0.0:
                # Khong match vector, skip
                continue
            
            # 3. SOFT ATTRIBUTE CHECK
            attribute_match = self._soft_check_attributes(
                person_id, metadata, current_attributes
            )
            
            if not attribute_match:
                # Attributes khong match (va confidence cao), skip
                continue
            
            # 4. TINH OVERALL SCORE
            # Temporal: 30% (neu gap lau thi less likely)
            temporal_score = max(0, 1 - (time_gap / config.TEMPORAL_MATCHING_WINDOW))
            
            # Vector: 70% (face/reid la main signal)
            overall_score = (temporal_score * 0.3) + (vector_score * 0.7)
            
            if overall_score > best_score:
                best_score = overall_score
                best_match = (person_id, overall_score, f"REMATCH ({vector_source})")
                
                logger.debug(
                    f"🔍 [REMATCH CANDIDATE] {person_id}\n"
                    f"   Temporal: {time_gap:.1f}s (score={temporal_score:.3f})\n"
                    f"   Vector: {vector_score:.3f} ({vector_source})\n"
                    f"   Overall: {overall_score:.3f}"
                )
        
        if best_match:
            person_id, score, source = best_match
            logger.info(
                f"✅ [CONFIRMED REMATCH] Track {track_id} → {person_id}\n"
                f"   Score: {score:.3f} | Source: {source}"
            )
            return best_match
        
        return None
    
    def check_llm_processing_needed(self, person_id: str) -> tuple:
        """
        Kiem tra xem co can gui LLM xu ly khong.
        
        Return: (need_llm, duration_minutes, reason)
        
        Logic:
        - Neu duration > 30 phut: can LLM (nguoi di lau roi quay lai)
        - Neu < 30 phut: khong can
        """
        metadata = self.db_manager.get_metadata(person_id)
        
        if not metadata:
            return (False, 0, "No metadata")
        
        first_seen_str = metadata.get('first_seen_time')
        last_seen_str = metadata.get('last_seen_time')
        
        if not first_seen_str or not last_seen_str:
            return (False, 0, "Missing time data")
        
        try:
            first_seen = datetime.fromisoformat(first_seen_str)
            last_seen = datetime.fromisoformat(last_seen_str)
            
            # Tinh duration (phut)
            duration = (last_seen - first_seen).total_seconds() / 60.0
            
            # Threshold: 30 phut
            if duration > 30:
                reason = f"Long absence: {duration:.1f} minutes"
                logger.info(f"🤖 [LLM FLAG] {person_id} - {reason}")
                return (True, duration, reason)
            else:
                return (False, duration, f"Recent: {duration:.1f} minutes")
                
        except Exception as e:
            logger.error(f"❌ [LLM CHECK] Error: {e}")
            return (False, 0, f"Error: {e}")
    
    def _get_all_confirmed_persons(self) -> dict:
        """
        Lay all confirmed persons tu metadata.
        Filter chi lay nhung co last_seen_time (khong can last_bbox)
        """
        confirmed_persons = {}
        
        # Iterate all person IDs in metadata
        all_metadata = self.db_manager.metadata or {}
        
        for person_id, meta in all_metadata.items():
            # Chi lay nhung co 'confirmed_gender' hoac 'confirmed_age' hoac 'confirmed_race'
            has_confirmed = any([
                'confirmed_gender' in meta,
                'confirmed_age' in meta,
                'confirmed_race' in meta
            ])
            
            if has_confirmed and 'last_seen_time' in meta:
                confirmed_persons[person_id] = meta
        
        return confirmed_persons
    
    def _soft_check_attributes(
        self,
        person_id: str,
        confirmed_metadata: dict,
        current_attributes: dict
    ) -> bool:
        """
        Soft check: Chi reject neu attributes thay doi RÕNG VA CONFIDENCE CAO
        
        Return: True neu attributes match hoac khong the kiem tra
        """
        
        # Lay confirmed attributes (immutable)
        confirmed_gender = confirmed_metadata.get('confirmed_gender', 'unknown')
        confirmed_age = confirmed_metadata.get('confirmed_age', 'unknown')
        confirmed_race = confirmed_metadata.get('confirmed_race', 'unknown')
        
        confirmed_gender_conf = confirmed_metadata.get('gender_confidence', 0.0)
        confirmed_age_conf = confirmed_metadata.get('age_confidence', 0.0)
        confirmed_race_conf = confirmed_metadata.get('race_confidence', 0.0)
        
        # Lay current attributes
        current_gender = current_attributes.get('gender', 'unknown')
        current_age = current_attributes.get('age_onnx', current_attributes.get('age', 'unknown'))
        current_race = current_attributes.get('race', 'unknown')
        
        current_gender_conf = current_attributes.get('gender_confidence', 0.0)
        current_age_conf = current_attributes.get('age_confidence', 0.0)
        current_race_conf = current_attributes.get('race_confidence', 0.0)
        
        # Helper function
        def is_invalid(val):
            return not val or str(val).lower() in ['unknown', 'n/a', 'chưa xác định', 'none', '']
        
        # SOFT RULES:
        # 1. Neu confirmed attribute khong chac chan (conf < 0.70) thi skip check
        # 2. Neu current attribute khong chac chan thi skip check
        # 3. Neu ca 2 deu chac chan thi phai match
        
        # CHECK GENDER
        if (not is_invalid(confirmed_gender) and confirmed_gender_conf >= 0.70 and
            not is_invalid(current_gender) and current_gender_conf >= 0.70):
            # Ca 2 deu chac chan -> phai match
            if confirmed_gender != current_gender:
                logger.warning(
                    f"❌ [REMATCH REJECT] {person_id} - Gender mismatch: "
                    f"{confirmed_gender} vs {current_gender}"
                )
                return False
        
        # CHECK AGE
        if (not is_invalid(confirmed_age) and confirmed_age_conf >= 0.70 and
            not is_invalid(current_age) and current_age_conf >= 0.70):
            # Ca 2 deu chac chan -> phai match
            if confirmed_age != current_age:
                logger.warning(
                    f"❌ [REMATCH REJECT] {person_id} - Age mismatch: "
                    f"{confirmed_age} vs {current_age}"
                )
                return False
        
        # CHECK RACE
        if (not is_invalid(confirmed_race) and confirmed_race_conf >= 0.70 and
            not is_invalid(current_race) and current_race_conf >= 0.70):
            # Ca 2 deu chac chan -> phai match
            if confirmed_race != current_race:
                logger.warning(
                    f"❌ [REMATCH REJECT] {person_id} - Race mismatch: "
                    f"{confirmed_race} vs {current_race}"
                )
                return False
        
        # All checks passed (hoac skipped due to low confidence)
        return True
