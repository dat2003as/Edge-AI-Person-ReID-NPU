# core/tracker/consolidation.py - FIXED VERSION

from collections import Counter
import numpy as np
import logging
import config
from .utils import TrackerUtils

logger = logging.getLogger(__name__)


class AttributeConsolidator:
    """
    🔥 FIXED VERSION - Xử lý gender/age/race với NULL-safe checking
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def consolidate(self, obj_data):
        """Consolidate attributes từ history và LOCK nếu đủ điều kiện"""
        final_id = obj_data.get('final_id')
        if not final_id: 
            return

        current_status = obj_data.get('status', 'pending')
        
        # Early return: Skip processing cho objects da fully locked (performance optimization)
        if current_status == 'confirmed' and obj_data.get('final_attributes'):
            db_meta = self.db_manager.get_metadata(final_id) if not final_id.startswith("Temp") else None
            if db_meta and db_meta.get('status') == 'confirmed':
                face_count = self.db_manager.count_vectors_for_id(config.FACE_NAMESPACE, final_id) if not final_id.startswith("Temp_") else 0
                if face_count >= config.MAX_FACE_VECTORS_PER_PROFILE:
                    return


        history_raw = list(obj_data.get('history_attributes', []))
        print(f"\n🔍 [CONSOLIDATE START] {final_id} | History: {len(history_raw)} frames")

        # 🔥 KHỞI TẠO final_attributes NẾU CHƯA CÓ
        if obj_data.get('final_attributes') is None:
            obj_data['final_attributes'] = {}

        # Lấy history hợp lệ
        history = [h for h in history_raw if h is not None and isinstance(h, dict)]
        if not history: 
            logger.warning(f"⏭️ [CONSOLIDATE] {final_id} - Empty history")
            return

        def is_valid(val):
            """Kiểm tra giá trị có hợp lệ không"""
            if val is None: 
                return False
            val_str = str(val).strip().lower()
            return val_str not in ['', 'n/a', 'none', 'chưa xác định', 'unknown', 'null']

        # ============================================================
        # BƯỚC 1: KIỂM TRA DB
        # ============================================================
        current_status = obj_data.get('status')

        db_meta = self.db_manager.get_metadata(final_id) if not final_id.startswith("Temp") else None
        is_metadata_stable = (
            db_meta is not None and 
            db_meta.get('status') == 'confirmed' and
            is_valid(db_meta.get('age')) and
            is_valid(db_meta.get('gender')) and
            is_valid(db_meta.get('race'))
        )

        # 🔥 KIỂM TRA SỐ LƯỢNG VECTORS
        face_vector_count = 0
        if not final_id.startswith("Temp_"):
            face_vector_count = self.db_manager.count_vectors_for_id(config.FACE_NAMESPACE, final_id)
            
        has_enough_vectors = (face_vector_count >= config.MAX_FACE_VECTORS_PER_PROFILE)

        is_fully_locked = is_metadata_stable and (face_vector_count >= config.MAX_FACE_VECTORS_PER_PROFILE)
    
        # 🔥 SKIP NẾU ĐÃ FULLY LOCKED
        if is_fully_locked and current_status == 'confirmed':
            logger.debug(f"⏭️ [CONSOLIDATE SKIP] {final_id} already fully locked")
            return


        # ============================================================
        # BƯỚC 2: XỬ LÝ AGE/GENDER/RACE - CONSOLIDATE + LOCK
        # ============================================================
        
        # 🔥 PRIORITY: CCCD DATA (TUYỆT ĐỐI)
        is_cccd = db_meta.get('cccd_matched') if db_meta else obj_data.get('cccd_matched')
        
        if is_cccd:
            # Lấy data từ DB hoặc Memory
            c_name = db_meta.get('cccd_name') if db_meta else obj_data.get('cccd_name')
            c_age = db_meta.get('cccd_age') if db_meta else obj_data.get('cccd_age')
            c_gender = db_meta.get('cccd_gender') if db_meta else obj_data.get('cccd_gender')
            c_country = db_meta.get('cccd_country') if db_meta else obj_data.get('cccd_country')
            
            obj_data['final_attributes'].update({
                'name': c_name,
                'gender': c_gender,
                'gender_confidence': 1.0,
                'age_onnx': str(c_age) if c_age is not None and c_age != 'unknown' else 'Unknown', # Handle None
                'age': str(c_age) if c_age is not None and c_age != 'unknown' else 'Unknown',
                'age_confidence': 1.0,
                'race': c_country,      # Display country as race/origin
                'race_confidence': 1.0,
                'country': c_country
            })
            # Ensure status confirmed
            if obj_data['status'] != 'confirmed':
                obj_data['status'] = 'confirmed'
                
            print(f"🔒 [CCCD LOCKED] {final_id} | Using CCCD Age: {c_age}, Gender: {c_gender}")

        elif is_metadata_stable:
            # Metadata đã locked (nhưng ko phải CCCD) → dùng DB
            obj_data['final_attributes'].update({
                'gender': db_meta.get('gender'),
                'gender_confidence': 1.0,
                'age': db_meta.get('age'),
                'age_confidence': 1.0,
                'race': db_meta.get('race'),
                'race_confidence': 1.0,
                'age_onnx': db_meta.get('age_onnx')
            })
            
            # 🔥 FIX: Cập nhật status sang 'confirmed' khi DB đã stable
            if obj_data['status'] != 'confirmed':
                obj_data['status'] = 'confirmed'
                logger.info(f"✅ [STATUS CHANGE] {final_id}: identified → confirmed (DB stable)")
            
            print(f"🔒 [METADATA STABLE] {final_id} | Vectors: {face_vector_count}/{config.MAX_FACE_VECTORS_PER_PROFILE}")
            
        else:
            # ============================================================
            # BƯỚC 2A: TÌM BEST FRAMES VỚI CONFIDENCE CAO NHẤT
            # ============================================================
            best_frames = {
                'gender': None,
                'race': None,
                'age_onnx': None
            }
            latest_idx = len(history) - 1

            for idx, result in enumerate(history):
                if not isinstance(result, dict): 
                    continue
                
                # --- GENDER ---
                g_data = result.get('gender_analysis')
                
                # 🔥 FIX: NULL-SAFE CHECKING
                if isinstance(g_data, dict):
                    g_val = g_data.get('gender')
                    g_conf = g_data.get('confidence', 0.0)
                    g_source = g_data.get('source', 'live')
                    
                    # Log debug cho frame cuối cùng
                    if idx == latest_idx:
                        print(f"   🔍 [LAST FRAME GENDER] Val: {g_val} | Conf: {g_conf} | Source: {g_source}")
                    
                    # Skip database source
                    if g_source == 'database': 
                        continue
                    
                    # 🔥 KIỂM TRA KỸ: Phải có cả value VÀ confidence
                    if is_valid(g_val) and g_conf > 0:
                        if not best_frames['gender'] or g_conf > best_frames['gender']['confidence']:
                            best_frames['gender'] = {
                                'value': g_val, 
                                'confidence': g_conf
                            }
                            print(f"      → Updated best gender: {g_val} (conf={g_conf:.3f})")

                # --- AGE + RACE (TỪ ONNX) ---
                r = result.get('age_race_analysis')
                if isinstance(r, dict):
                    r_source = r.get('source', 'live')
                    
                    if r_source == 'database':
                        continue
                    
                    # Race
                    if is_valid(r.get('race')):
                        conf_r = r.get('race_conf', 0)
                        
                        if conf_r > 0 and (not best_frames['race'] or conf_r > best_frames['race']['confidence']):
                            best_frames['race'] = {
                                'value': r['race'], 
                                'confidence': conf_r
                            }
                    
                    # Age
                    if is_valid(r.get('age_onnx')):
                        conf_a = r.get('age_onnx_conf', 0)
                        
                        if conf_a > 0 and (not best_frames['age_onnx'] or conf_a > best_frames['age_onnx']['confidence']):
                            best_frames['age_onnx'] = {
                                'value': r['age_onnx'], 
                                'confidence': conf_a
                            }

            # 🔥 DEBUG: Log để verify
            print(f"   📊 [BEST FRAMES SUMMARY]")
            if best_frames['gender']:
                print(f"      - Gender: {best_frames['gender']['value']} (conf={best_frames['gender']['confidence']:.3f})")
            else:
                print(f"      - Gender: NONE")
            
            if best_frames['age_onnx']:
                print(f"      - Age: {best_frames['age_onnx']['value']} (conf={best_frames['age_onnx']['confidence']:.3f})")
            else:
                print(f"      - Age: NONE")
            
            if best_frames['race']:
                print(f"      - Race: {best_frames['race']['value']} (conf={best_frames['race']['confidence']:.3f})")
            else:
                print(f"      - Race: NONE")

            # --- GÁN GIÁ TRỊ VỚI CONFIDENCE FILTER ---
            # 🔥 CHỈ HIỂN THỊ KHI CONF ĐỦ CAO (tránh hiển thị sai)
            MIN_DISPLAY_CONF_GENDER = 0.65  # Gender cần 80%+
            MIN_DISPLAY_CONF_AGE = 0.8     # Age cần 55%+
            MIN_DISPLAY_CONF_RACE = 0.80    # Race cần 80%+
            
            # Gender
            if best_frames['gender'] and best_frames['gender']['confidence'] >= MIN_DISPLAY_CONF_GENDER:
                final_gender = best_frames['gender']['value']
                final_gender_conf = best_frames['gender']['confidence']
            else:
                final_gender = "unknown"
                final_gender_conf = 0.0
            
            # Race
            if best_frames['race'] and best_frames['race']['confidence'] >= MIN_DISPLAY_CONF_RACE:
                final_race = best_frames['race']['value']
                final_race_conf = best_frames['race']['confidence']
            else:
                final_race = "unknown"
                final_race_conf = 0.0
            
            # Age
            if best_frames['age_onnx'] and best_frames['age_onnx']['confidence'] >= MIN_DISPLAY_CONF_AGE:
                final_age_onnx = best_frames['age_onnx']['value']
                final_age_onnx_conf = best_frames['age_onnx']['confidence']
            else:
                final_age_onnx = "unknown"
                final_age_onnx_conf = 0.0

            obj_data['final_attributes'].update({
                'gender': final_gender,
                'gender_confidence': final_gender_conf,
                'race': final_race,
                'race_confidence': final_race_conf,
                'age_onnx': final_age_onnx,
                'age_confidence': final_age_onnx_conf
            })
            
            print(f"   ✅ [FINAL VALUES WITH CONFIDENCE FILTER]")
            print(f"      - Gender: {final_gender} (conf={final_gender_conf:.3f}) {'✓' if final_gender_conf >= MIN_DISPLAY_CONF_GENDER else '✗ TOO LOW'}")
            print(f"      - Age: {final_age_onnx} (conf={final_age_onnx_conf:.3f}) {'✓' if final_age_onnx_conf >= MIN_DISPLAY_CONF_AGE else '✗ TOO LOW'}")
            print(f"      - Race: {final_race} (conf={final_race_conf:.3f}) {'✓' if final_race_conf >= MIN_DISPLAY_CONF_RACE else '✗ TOO LOW'}")
            
            # ============================================================
            # BƯỚC 2B: 🔥 KIỂM TRA ĐIỀU KIỆN CHỐT (LOCK)
            # ============================================================
            STANDARD_CONF_GENDER = 0.70  # Gender dễ dàng hơn
            STANDARD_CONF_AGE = 0.80
            STANDARD_CONF_RACE = 0.80
            HIGH_CONF = 0.85
            FAST_CONF = 0.9
            MIN_FRAMES_FAST = 1
            MIN_FRAMES_STANDARD = 2
            MIN_FRAMES_PROGRESSIVE = 10

            # 🔥 NULL-SAFE checking
            gender_standard = (final_gender_conf >= STANDARD_CONF_GENDER and final_gender != 'unknown')
            gender_high = (final_gender_conf >= HIGH_CONF and final_gender != 'unknown')
            gender_fast = (final_gender_conf >= FAST_CONF and final_gender != 'unknown')

            age_standard = (final_age_onnx_conf >= STANDARD_CONF_AGE and final_age_onnx != 'unknown')
            age_high = (final_age_onnx_conf >= 0.85 and final_age_onnx != 'unknown')
            age_fast = (final_age_onnx_conf >= 0.9 and final_age_onnx != 'unknown')

            race_standard = (final_race_conf >= STANDARD_CONF_RACE and final_race != 'unknown')
            race_high = (final_race_conf >= HIGH_CONF and final_race != 'unknown')
            race_fast = (final_race_conf >= FAST_CONF and final_race != 'unknown')

            history_len = len(history)
            
            # 🔥 MODE 1: FAST (Tất cả đều cao)
            fast_trigger = (
                gender_high and      
                age_high and          
                race_high and        
                history_len >= MIN_FRAMES_FAST    
            )

            # 🔥 MODE 2: STANDARD
            standard_trigger = (
                history_len >= MIN_FRAMES_STANDARD and
                gender_standard and   
                age_standard and      
                race_standard         
            )
            
            # 🔥 MODE 3: PROGRESSIVE (Cả 3 STANDARD + ít nhất 1 HIGH)
            progressive_trigger = (
                history_len >= MIN_FRAMES_PROGRESSIVE and
                gender_standard and
                age_standard and
                race_standard and
                sum([bool(gender_high), bool(age_high), bool(race_high)]) >= 1
            )

            # 🔥 ĐK LOCK DB: CẦN ĐỦ 3 ATTRIBUTES (Gender + Age + Race)
            # ⚠️ KHÔNG bao gồm gender_only_fast (dùng riêng cho SEND-2 LLM)
            metadata_ready = fast_trigger or standard_trigger or progressive_trigger

            print(
                f"🔍 [LOCK CHECK] {final_id} | Frames: {history_len}\n"
                f"   Gender: {final_gender} (Conf: {final_gender_conf:.2f}) "
                f"[STD:{gender_standard} HIGH:{gender_high} FAST:{gender_fast}]\n"
                f"   Age:    {final_age_onnx} (Conf: {final_age_onnx_conf:.2f}) "
                f"[STD:{age_standard} HIGH:{age_high} FAST:{age_fast}]\n"
                f"   Race:   {final_race} (Conf: {final_race_conf:.2f}) "
                f"[STD:{race_standard} HIGH:{race_high} FAST:{race_fast}]\n"
                f"   → Fast: {fast_trigger} | Standard: {standard_trigger} | Progressive: {progressive_trigger}"
            )

            # ============================================================
            # BƯỚC 2C: 🔥 LOCK DATABASE NẾU READY
            # ============================================================
            if metadata_ready and not final_id.startswith("Temp_"):
                trigger_mode = (
                    "FAST ⚡" if fast_trigger else
                    "STANDARD ⏳" if standard_trigger else
                    "PROGRESSIVE 📈" if progressive_trigger else
                    "UNKNOWN"
                )
                
                lock_data = {
                    'status': 'confirmed',
                    'gender': final_gender,
                    'age': final_age_onnx,
                    'race': final_race,
                    'age_onnx': final_age_onnx
                }
                
                if self.db_manager.update_metadata(final_id, lock_data):
                    obj_data['status'] = 'confirmed'
                    obj_data['locked_at_frame'] = obj_data.get('last_analysis_frame', 0)
                    
                    print(f"✨ [METADATA LOCKED] {trigger_mode} cho {final_id}")
                    print(f"   ✅ Gender={final_gender} | Age={final_age_onnx} | Race={final_race}")
                    print(f"   📊 Frames: {history_len} | Vectors: {face_vector_count}/{config.MAX_FACE_VECTORS_PER_PROFILE}")
                    
                    self.db_manager.save_all_databases()
                else:
                    logger.error(f"❌ [METADATA LOCK FAILED] {final_id}")
                
                if has_enough_vectors:
                    print(f"   ✅ [READY FOR SEND-2] Đã đủ cả metadata + vectors!")
                else:
                    print(f"   ⏳ [COLLECTING VECTORS] Cần thêm {config.MAX_FACE_VECTORS_PER_PROFILE - face_vector_count} vectors cho SEND-2")
            
            elif not metadata_ready:
                logger.info(
                    f"⏳ [WAITING] {final_id} - Chưa đủ điều kiện lock\n"
                    f"   Frames: {history_len}\n"
                    f"   Gender: FAST={gender_fast} / STD={gender_standard} (conf={final_gender_conf:.2f})\n"
                    f"   Age:    FAST={age_fast} / STD={age_standard} (conf={final_age_onnx_conf:.2f})\n"
                    f"   Race:   FAST={race_fast} / STD={race_standard} (conf={final_race_conf:.2f})"
                )
        
        # ============================================================
        # BƯỚC 3: XỬ LÝ EMOTION & CLOTHING (ĐỘNG)
        # ============================================================
        latest_res = history[-1] if history else {}
        latest_clothing = latest_res.get('clothing_analysis') or {}
        latest_emotion = latest_res.get('emotion_analysis') or {}

        votes_dynamic = {'emotion': [], 'upper_type': [], 'lower_type': []}
        upper_colors, lower_colors, skin_colors_bgr = [], [], []

        for i, result in enumerate(history):
            if not isinstance(result, dict):
                continue

            # EMOTION
            emo = result.get('emotion_analysis') or {}
            if isinstance(emo, dict):
                val_emo = emo.get('emotion')
                if val_emo: 
                    votes_dynamic['emotion'].append(str(val_emo))
            elif emo:
                votes_dynamic['emotion'].append(str(emo))

            # CLOTHING
            c_res = result.get('clothing_analysis') or {}
            if c_res:
                sleeve = c_res.get('sleeve_type')
                pants = c_res.get('pants_type')
                
                if sleeve and sleeve not in ['Chưa xác định', 'Chua xac dinh']: 
                    votes_dynamic['upper_type'].append(str(sleeve))
                
                if pants and pants not in ['Chưa xác định', 'Chua xac dinh']: 
                    votes_dynamic['lower_type'].append(str(pants))
                
                if c_res.get('skin_tone_bgr') is not None: 
                    skin_colors_bgr.append(c_res['skin_tone_bgr'])
                
                raw = c_res.get('raw_colors') or {}
                if raw.get('brachium_colors'): 
                    upper_colors.extend([
                        c['bgr'] for c in raw['brachium_colors'] 
                        if isinstance(c, dict) and 'bgr' in c
                    ])
                if raw.get('thigh_colors'): 
                    lower_colors.extend([
                        c['bgr'] for c in raw['thigh_colors'] 
                        if isinstance(c, dict) and 'bgr' in c
                    ])

        # --- EMOTION SMART DISPLAY ---
        # 🔥 Chỉ update khi confidence cao, còn không giữ giá trị tốt nhất
        latest_emo = latest_emotion.get('emotion') if isinstance(latest_emotion, dict) else latest_emotion
        latest_emo_conf = latest_emotion.get('confidence', 0) if isinstance(latest_emotion, dict) else 0
        
        MIN_EMO_CONF = 0.6  # Ngưỡng tin cậy tối thiểu
        
        # Lấy emotion hiện tại (nếu có)
        current_emo = obj_data['final_attributes'].get('emotion', 'N/A')
        current_emo_conf = obj_data['final_attributes'].get('emotion_confidence', 0)
        
        # Chỉ update nếu emotion mới TỐT HƠN hoặc chưa có emotion
        if latest_emo and latest_emo_conf >= MIN_EMO_CONF:
            if latest_emo_conf > current_emo_conf or current_emo == 'N/A':
                obj_data['final_attributes']['emotion'] = str(latest_emo)
                obj_data['final_attributes']['emotion_confidence'] = latest_emo_conf
                print(f"   ✅ [EMOTION UPDATE] {latest_emo} (conf={latest_emo_conf:.2f})")
        elif current_emo == 'N/A':
            # Chưa có emotion nào → dùng voting từ history
            if votes_dynamic['emotion']:
                try:
                    votes_dynamic['emotion'] = [str(v) for v in votes_dynamic['emotion'] if v]
                    if votes_dynamic['emotion']:
                        obj_data['final_attributes']['emotion'] = Counter(votes_dynamic['emotion']).most_common(1)[0][0]
                        obj_data['final_attributes']['emotion_confidence'] = 0.5
                except:
                    pass

        # --- CLOTHING TYPE SMART DISPLAY ---
        # 🔥 Chỉ update khi có kết quả rõ ràng, còn không giữ giá trị tốt
        for attr, vote_key, latest_key in [
            ('upper_type', 'upper_type', 'sleeve_type'), 
            ('lower_type', 'lower_type', 'pants_type')
        ]:
            # Lấy giá trị latest
            latest_val = latest_clothing.get(latest_key)
            is_latest_valid = (latest_val and latest_val not in ['Chưa xác định', 'Chua xac dinh', 'unknown'])
            
            # Lấy giá trị hiện tại (đang hiển thị)
            current_val = obj_data['final_attributes'].get(attr, 'Chua xac dinh')
            is_current_valid = (current_val not in ['Chưa xác định', 'Chua xac dinh', 'unknown'])
            
            # Logic update: Ưu tiên giữ giá trị tốt
            if is_latest_valid:
                # Latest có giá trị hợp lệ → update
                obj_data['final_attributes'][attr] = str(latest_val)
            elif is_current_valid:
                # Latest không có nhưng current vẫn tốt → giữ nguyên
                pass  # Không làm gì, giữ current_val
            elif votes_dynamic[vote_key]:
                # Cả 2 đều không có → dùng voting
                try:
                    valid_votes = [
                        str(v) for v in votes_dynamic[vote_key] 
                        if v and v not in ['Chưa xác định', 'Chua xac dinh', 'unknown']
                    ]
                    if valid_votes:
                        obj_data['final_attributes'][attr] = Counter(valid_votes).most_common(1)[0][0]
                    else:
                        obj_data['final_attributes'][attr] = "Chua xac dinh"
                except:
                    obj_data['final_attributes'][attr] = "Chua xac dinh"
            else:
                obj_data['final_attributes'][attr] = "Chua xac dinh"

        # --- COLOR AGGREGATION with SMART DISPLAY ---
        if skin_colors_bgr:
            obj_data['final_attributes']['skin_tone_bgr'] = np.mean(skin_colors_bgr, axis=0).astype(int).tolist()
        
        # 🔥 UPPER COLOR: Ưu tiên latest, fallback voting
        latest_upper_color = latest_clothing.get('upper_color')  # Đây là 1 giá trị [B,G,R]
        if latest_upper_color and isinstance(latest_upper_color, list) and len(latest_upper_color) == 3:
            # Latest có màu → dùng trực tiếp
            obj_data['final_attributes']['upper_color'] = latest_upper_color
            print(f"   🎨 [UPPER COLOR] From latest: {latest_upper_color}")
        elif upper_colors:
            # Không có latest → dùng voting từ history
            dom_u = TrackerUtils.find_dominant_color(upper_colors)
            if dom_u: 
                obj_data['final_attributes']['upper_color'] = dom_u[::-1]
                print(f"   🎨 [UPPER COLOR] From voting: {dom_u[::-1]}")
        
        # 🔥 LOWER COLOR: Ưu tiên latest, fallback voting
        latest_lower_color = latest_clothing.get('lower_color')  # Đây là 1 giá trị [B,G,R]
        if latest_lower_color and isinstance(latest_lower_color, list) and len(latest_lower_color) == 3:
            # Latest có màu → dùng trực tiếp
            obj_data['final_attributes']['lower_color'] = latest_lower_color
            print(f"   🎨 [LOWER COLOR] From latest: {latest_lower_color}")
        elif lower_colors:
            # Không có latest → dùng voting từ history
            dom_l = TrackerUtils.find_dominant_color(lower_colors)
            if dom_l: 
                obj_data['final_attributes']['lower_color'] = dom_l[::-1]
                print(f"   🎨 [LOWER COLOR] From voting: {dom_l[::-1]}")
    # 🔥 ==================== SEND-2 TRIGGER ====================
    def trigger_send_2_if_ready(self, obj_data, person_id, dual_stream_manager, llm_sender):
        """
        🔥 GỬI SEND-2 KHI AI ĐÃ CÓ KẾT QUẢ ỔN ĐỊNH (CHƯA CẦN CONFIRMED)
        
        Điều kiện:
        1. SEND-1 đã hoàn tất
        2. Có đủ dữ liệu AI (gender + age + race) với confidence cao
        3. Chưa gửi SEND-2
        
        ✅ KHÔNG CẦN: metadata locked hoặc đủ 25 vectors
        """
        if not person_id or person_id.startswith('Temp_'):
            return False
        
        if not dual_stream_manager or not llm_sender:
            return False
        
        # ✅ CHECK 1: SEND-1 đã gửi?
        state = dual_stream_manager.get_state(person_id)
        if not state or not state['send_1_done']:
            logger.debug(f"⏳ [SEND-2] {person_id} waiting for SEND-1")
            return False
        
        # ✅ CHECK 2: SEND-2 chưa gửi?
        if state['send_2_done']:
            logger.debug(f"⏭️ [SEND-2] {person_id} already sent")
            return False
        
        # ✅ CHECK 3: Có dữ liệu AI từ consolidation không?
        final_attrs = obj_data.get('final_attributes')
        if not final_attrs:
            logger.debug(f"⏳ [SEND-2] {person_id} chưa có final_attributes")
            return False
        
        # 🔥 KIỂM TRA CHẤT LƯỢNG DỮ LIỆU AI
        def is_valid(val):
            return val and str(val).lower() not in ['unknown', 'n/a', 'chưa xác định', 'none', '']
        
        gender = final_attrs.get('gender')
        age = final_attrs.get('age_onnx', final_attrs.get('age'))
        race = final_attrs.get('race')

        # 🔥 TÌM CONFIDENCE CAO NHẤT CHO MỖI THUỘC TÍNH (từ history)
        history = obj_data.get('history_attributes', [])
        best_gender_conf = 0
        best_age_conf = 0
        best_race_conf = 0
        
        for result in history:
            if not isinstance(result, dict):
                continue
            # Gender
            g = result.get('gender_analysis')
            if isinstance(g, dict) and g.get('source') != 'database':
                best_gender_conf = max(best_gender_conf, g.get('confidence', 0))
            # Age/Race
            ar = result.get('age_race_analysis')
            if isinstance(ar, dict) and ar.get('source') != 'database':
                best_age_conf = max(best_age_conf, ar.get('age_onnx_conf', 0))
                best_race_conf = max(best_race_conf, ar.get('race_conf', 0))

        # 🔥 ĐẶC BIỆT: NẾU GENDER RẤT RÕ (>0.85) THÌ GỬI LUÔN, KHÔNG CẦN VALID COUNT
        is_gender_fast_trigger = (is_valid(gender) and best_gender_conf >= 0.85)
        
        # Tối thiểu phải có 2/3 thuộc tính hợp lệ (NẾU KHÔNG PHẢI GENDER FAST)
        valid_count = sum([is_valid(gender), is_valid(age), is_valid(race)])
        
        if valid_count < 2 and not is_gender_fast_trigger:
            logger.debug(f"⏳ [SEND-2] {person_id} AI data chưa đủ: gender={gender}, age={age}, race={race}")
            return False
        
        if not is_gender_fast_trigger:
            if len(history) < 5:  # Tối thiểu 5 frames
                logger.debug(f"⏳ [SEND-2] {person_id} chỉ có {len(history)} frames, cần ít nhất 5")
                return False
        
        # 🔥 KIỂM TRA NGƯỠNG CONFIDENCE (nới lỏng hơn so với LOCK)
        min_conf_threshold = 0.70  # Thấp hơn 0.80 của lock
        
        confidence_ok = (
            (is_valid(gender) and best_gender_conf >= min_conf_threshold) or
            (is_valid(age) and best_age_conf >= 0.75) or
            (is_valid(race) and best_race_conf >= min_conf_threshold)
        )
        
        if not confidence_ok and not is_gender_fast_trigger:
             logger.debug(f"⏳ [SEND-2] {person_id} confidence chưa đủ: G={best_gender_conf:.2f}, A={best_age_conf:.2f}, R={best_race_conf:.2f}")
             return False
        
        if is_gender_fast_trigger:
             logger.info(f"🚀 [SEND-2 FAST] {person_id} triggered by HIGH CONFIDENCE GENDER ({gender}, {best_gender_conf:.2f})")
        
        # 🔥 CHUẨN BỊ DỮ LIỆU GỬI
        ai_data = {
            'age': age if is_valid(age) else 'unknown',
            'gender': gender if is_valid(gender) else 'unknown',
            'race': race if is_valid(race) else 'unknown'
        }
        
        # 🔥 GỬI SEND-2
        success = dual_stream_manager.send_second_with_ai(person_id, ai_data, llm_sender)
        
        if success:
            logger.info(
                f"✅ [SEND-2] AI data sent for {person_id} (UNLOCKED)\n"
                f"   Age={ai_data['age']} (conf={best_age_conf:.2f})\n"
                f"   Gender={ai_data['gender']} (conf={best_gender_conf:.2f})\n"
                f"   Race={ai_data['race']} (conf={best_race_conf:.2f})\n"
                f"   Valid attributes: {valid_count}/3 | History frames: {len(history)}"
            )
            return True
        else:
            logger.error(f"❌ [SEND-2] Failed to send for {person_id}")
            return False