# vector_database.py - FIXED VERSION
import os
import pickle
import faiss
import numpy as np
import threading
import time
import logging
import copy
from collections import defaultdict
from typing import Optional, List, Tuple
import config

logger = logging.getLogger(__name__)

class VectorDatabase_Manager:
    """
    Quản lý cơ sở dữ liệu vector Faiss.
    - Hỗ trợ lưu nhiều vector cho một ID.
    - Sử dụng cơ chế bỏ phiếu kết hợp, ưu tiên ID có điểm trung bình cao nhất
    - 🔥 HỖ TRỢ DYNAMIC NAMESPACES + TỰ ĐỘNG TẠO DB MỚI
    """
    def __init__(self, index_dir="faiss_indexes"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        
        # ============================================================
        # 🔥 PHẦN 1: KHỞI TẠO LOCKS, STRUCTURES & METADATA TRƯỚC
        # ============================================================
        self.db_lock = threading.Lock()
        self.dimensions = {}
        self.indexes = {}
        self.id_maps = {}
        self.has_unsaved_changes = {}
        
        # ✅ PHẢI KHỞI TẠO METADATA TRƯỚC KHI GỌI BẤT KỲ HÀM SAVE NÀO
        self.metadata = self._load_metadata()
        
        # ============================================================
        # 🔥 PHẦN 2: LOAD HOẶC TẠO MỚI NAMESPACES
        # ============================================================
        existing_namespaces = self._discover_namespaces()
        
        if existing_namespaces:
            print(f"📂 Phát hiện {len(existing_namespaces)} namespaces: {existing_namespaces}")
            for ns in existing_namespaces:
                self._load_namespace(ns)
        else:
            print(f"🆕 Database mới tại '{self.index_dir}', đang khởi tạo default namespaces...")
            # Hàm này gọi save_all_databases(), giờ đã an toàn vì metadata đã tồn tại
            self._init_default_namespaces()
        
        # ============================================================
        # PHẦN 3: CẬP NHẬT FLAGS
        # ============================================================
        for ns in self.indexes.keys():
            if ns not in self.has_unsaved_changes:
                self.has_unsaved_changes[ns] = False
        
        print(f"✅ Khởi tạo Faiss Vector DB Manager ({self.index_dir}) thành công.")
    
    def _discover_namespaces(self) -> List[str]:
        """
        🔥 TỰ ĐỘNG PHÁT HIỆN NAMESPACES TỪ FILES
        Tìm tất cả .index files trong thư mục
        """
        namespaces = []
        
        if not os.path.exists(self.index_dir):
            return namespaces
        
        for filename in os.listdir(self.index_dir):
            if filename.endswith('.index'):
                # Lấy tên namespace từ filename
                # Ví dụ: "face_features.index" → "face_features"
                ns = filename[:-6]  # Bỏ ".index"
                namespaces.append(ns)
        
        return namespaces
    
    def _init_default_namespaces(self):
        """
        🔥 KHỞI TẠO DEFAULT NAMESPACES KHI DB TRỐNG
        Tạo các namespace cơ bản từ config
        """
        # Xác định default namespaces dựa vào index_dir
        if "cccd" in self.index_dir.lower():
            # Database CCCD: Chỉ cần Face namespace
            default_namespaces = {
                "CCCD_FACES": config.FACE_VECTOR_DIM
            }
            print("   📋 CCCD Database: Tạo namespace CCCD_FACES")
        else:
            # Database Tracking: Cần ReID + Face
            default_namespaces = {
                config.REID_NAMESPACE: config.OSNET_VECTOR_DIM,
                config.FACE_NAMESPACE: config.FACE_VECTOR_DIM
            }
            print(f"   📋 Tracking Database: Tạo namespaces {list(default_namespaces.keys())}")
        
        # Tạo các namespace
        self.dimensions = default_namespaces
        
        for ns in self.dimensions:
            self.indexes[ns] = self._create_new_index(ns)
            self.id_maps[ns] = []
            print(f"      ✅ Khởi tạo '{ns}' (dim={self.dimensions[ns]})")
        
        # 🔥 LƯU NGAY LẬP TỨC ĐỂ TẠO FILES
        print("   💾 Lưu database mới...")
        self.save_all_databases()
    
    def _load_namespace(self, namespace: str):
        """
        🔥 LOAD 1 NAMESPACE TỪ FILES
        Tự động detect dimension từ index file
        """
        index_path, id_map_path = self._get_paths(namespace)
        
        # Load index
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            self.indexes[namespace] = index
            self.dimensions[namespace] = index.d  # Lấy dimension từ index
            print(f"   ✅ Loaded '{namespace}': {index.ntotal} vectors, dim={index.d}")
        else:
            print(f"   ⚠️ Index file not found: {index_path}")
            return
        
        # Load id_map
        if os.path.exists(id_map_path) and os.path.getsize(id_map_path) > 0:
            try:
                with open(id_map_path, 'rb') as f:
                    self.id_maps[namespace] = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print(f"⚠️ ID map '{id_map_path}' lỗi. Khởi tạo danh sách trống.")
                self.id_maps[namespace] = []
        else:
            print(f"   ⚠️ ID map not found hoặc rỗng: {id_map_path}, khởi tạo mới")
            self.id_maps[namespace] = []
    
    def _create_new_index(self, namespace: str) -> faiss.Index:
        """Tạo index mới cho namespace"""
        if namespace not in self.dimensions:
            raise ValueError(f"Dimension not defined for namespace '{namespace}'")
        
        dim = self.dimensions[namespace]
        return faiss.IndexFlatIP(dim)
    
    def _get_paths(self, namespace: str) -> Tuple[str, str]:
        """Lấy đường dẫn index và id_map"""
        index_path = os.path.join(self.index_dir, f"{namespace}.index")
        id_map_path = os.path.join(self.index_dir, f"{namespace}.pkl")
        return index_path, id_map_path
    
    def _get_metadata_path(self):
        return os.path.join(self.index_dir, "metadata.pkl")
    
    def _load_metadata(self):
        path = self._get_metadata_path()
        # Kiểm tra file tồn tại và có dung lượng lớn hơn 0
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print(f"⚠️ Metadata file '{path}' bị lỗi hoặc rỗng. Khởi tạo mới.")
                return {}
        return {}
    
    def _save_data(self, namespace: str):
        """Lưu 1 namespace"""
        with self.db_lock:
            index_path, id_map_path = self._get_paths(namespace)
            if self.indexes[namespace] and self.has_unsaved_changes.get(namespace, False):
                faiss.write_index(self.indexes[namespace], index_path)
                with open(id_map_path, 'wb') as f:
                    pickle.dump(self.id_maps[namespace], f)
                print(f"Đã lưu index và ID map cho namespace '{namespace}'.")
                self.has_unsaved_changes[namespace] = False
    
    def save_all_databases(self):
        """Lưu tất cả namespaces"""
        print("💾 [DB] Đang thực hiện lưu dữ liệu...")
        with self.db_lock:
            for ns in self.indexes.keys():
                index_path, id_map_path = self._get_paths(ns)
                if self.indexes[ns] is not None:
                    faiss.write_index(self.indexes[ns], index_path)
                    with open(id_map_path, 'wb') as f:
                        pickle.dump(self.id_maps[ns], f)
            
            # Lưu Metadata
            meta_path = self._get_metadata_path()
            with open(meta_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        print("✅ [DB] Lưu thành công!")
    
    def add_vectors(self, namespace: str, vector_id: str, vectors_data: List[list]):
        """Thêm vectors cho một ID"""
        if not vectors_data:
            return
        
        # 🔥 KIỂM TRA NAMESPACE TỒN TẠI
        if namespace not in self.indexes:
            print(f"⚠️ Namespace '{namespace}' không tồn tại. Namespaces hiện có: {list(self.indexes.keys())}")
            return
        
        with self.db_lock:
            vectors_np = np.array(vectors_data, dtype='float32')
            faiss.normalize_L2(vectors_np)
            self.indexes[namespace].add(vectors_np)
            self.id_maps[namespace].extend([vector_id] * len(vectors_data))
            self.has_unsaved_changes[namespace] = True
    
    def search_vector_with_voting(self, namespace: str, query_vector: list) -> Optional[Tuple[str, float]]:
        """Tìm kiếm vector với voting mechanism"""
        # 🔥 KIỂM TRA NAMESPACE TỒN TẠI
        if namespace not in self.indexes:
            print(f"⚠️ Search failed: Namespace '{namespace}' không tồn tại")
            return None
        
        index = self.indexes[namespace]
        
        # Kiểm tra DB trống
        if index.ntotal == 0:
            return None
        
        # Lấy cấu hình threshold
        if namespace == config.FACE_NAMESPACE or namespace == "CCCD_FACES" or namespace == "face":
            sim_threshold = config.FACE_DB_SEARCH_SIMILARITY_THRESHOLD
            min_votes_cfg = config.FACE_MIN_VOTES_FOR_MATCH
        else:
            sim_threshold = config.REID_DB_SEARCH_SIMILARITY_THRESHOLD
            min_votes_cfg = config.REID_MIN_VOTES_FOR_MATCH
        
        # Tìm kiếm Top-K
        query_np = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_np)
        distances, indices = index.search(query_np, config.SEARCH_TOP_K)
        
        print(f"\n🔍 [DB-Search] Namespace: {namespace} (Tổng: {index.ntotal} vector)")
        
        # Thu thập ứng viên
        candidates_data = defaultdict(list)
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            score = float(distances[0][i])
            match_id = self.id_maps[namespace][idx]
            
            if score >= sim_threshold:
                candidates_data[match_id].append(score)
            
            status = "✅" if score >= sim_threshold else "❌"
            print(f"   - {status} ID: {match_id:<12} | Score: {score:.4f}")
        
        # 🔥 Bỏ phiếu THÍCH ỨNG dựa trên SCORE
        finalists = []
        for mid, scores in candidates_data.items():
            vectors_in_db = self.count_vectors_for_id(namespace, mid)
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            combined_score = (avg_score * 0.7) + (max_score * 0.3)
            
            # 🔥 DYNAMIC THRESHOLD: Score cao → ít votes cần thiết
            if max_score >= config.DYNAMIC_MATCH_VERY_HIGH_THRESHOLD:  # >= 0.85
                adaptive_min_votes = config.DYNAMIC_MATCH_VERY_HIGH_MIN_VOTES  # 1 vote
            elif max_score >= config.DYNAMIC_MATCH_HIGH_THRESHOLD:  # >= 0.75
                adaptive_min_votes = config.DYNAMIC_MATCH_HIGH_MIN_VOTES  # 2 votes
            else:  # < 0.75
                adaptive_min_votes = config.DYNAMIC_MATCH_LOW_MIN_VOTES  # 3 votes
            
            # Fallback nếu DB không có đủ vectors
            adaptive_min_votes = min(adaptive_min_votes, vectors_in_db)
            
            if len(scores) >= adaptive_min_votes:
                finalists.append({
                    'id': mid,
                    'score': combined_score,
                    'votes': len(scores),
                    'in_db': vectors_in_db,
                    'max_score': max_score
                })
        
        if not finalists:
            print("   => ⚠️ Không ứng viên nào đạt ngưỡng Votes.")
            return None
        
        # Sắp xếp và trả về winner
        finalists.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"   => 🏆 CHIẾN THẮNG: {finalists[0]['id']} (Score: {finalists[0]['score']:.4f}, Votes: {finalists[0]['votes']})")
        return finalists[0]['id'], float(finalists[0]['score'])
    
    def count_vectors_for_id(self, namespace: str, vector_id: str) -> int:
        """Đếm số vector của một ID"""
        with self.db_lock:
            if namespace not in self.id_maps:
                return 0
            return self.id_maps[namespace].count(vector_id)
    
    def count_total_vectors(self, namespace: str) -> int:
        """
        🔥 MỚI: Đếm tổng số vectors trong namespace
        """
        with self.db_lock:
            if namespace not in self.indexes:
                return 0
            
            index = self.indexes[namespace]
            if index is None:
                return 0
            
            return index.ntotal
    
    def get_max_person_id(self) -> int:
        """Tìm số ID lớn nhất từ Person_X"""
        max_id = 0
        all_ids = set()
        
        for namespace in self.id_maps:
            all_ids.update(self.id_maps[namespace])
        
        for person_id in all_ids:
            if isinstance(person_id, str) and person_id.startswith("Person_"):
                try:
                    num = int(person_id.split('_')[1])
                    if num > max_id:
                        max_id = num
                except (ValueError, IndexError):
                    continue
        return max_id
    
    def update_metadata(self, person_id: str, attributes: dict):
        """Cập nhật metadata"""
        with self.db_lock:
            if person_id not in self.metadata:
                self.metadata[person_id] = {}
            self.metadata[person_id].update(attributes)
            return True
    
    def save_cccd_metadata(self, person_id: str, cccd_info: dict):
        """
        Lưu thông tin CCCD đầy đủ vào metadata
        Args:
            person_id: ID của người (Person_X)
            cccd_info: Dict chứa {name, age, gender, race, country, cccd_number}
        """
        with self.db_lock:
            if person_id not in self.metadata:
                self.metadata[person_id] = {}
            
            # Lưu CCCD info với flag matched=true
            self.metadata[person_id].update({
                'cccd_matched': True,
                'cccd_name': cccd_info.get('name', 'Unknown'),
                'cccd_age': cccd_info.get('age', 'Unknown'),
                'cccd_gender': cccd_info.get('gender', 'Unknown'),
                'cccd_race': cccd_info.get('race', 'Unknown'),
                'cccd_country': cccd_info.get('country', 'Unknown'),
                'cccd_number': cccd_info.get('cccd_number', 'Unknown'),
                'cccd_confidence': cccd_info.get('confidence', 0.0),
                'cccd_timestamp': time.time()
            })
            
            logger.info(f"[SAVE CCCD] {person_id}: {self.metadata[person_id].get('cccd_name')}")
            return True
    
    def get_metadata(self, person_id: str) -> dict:
        """Lấy metadata"""
        with self.db_lock:
            return self.metadata.get(person_id, {})

    def save_metadata(self):
        """
        🔥 LƯU METADATA-ONLY (Non-blocking I/O)
        Copy dữ liệu trong lock, sau đó ghi file ngoài lock để tránh lag UI
        """
        try:
            # 1. Snapshot data (Fast, RAM only)
            with self.db_lock:
                data_snapshot = copy.deepcopy(self.metadata)
            
            # 2. Write to disk (Slow, IO) - NO LOCK HERE
            meta_path = self._get_metadata_path()
            with open(meta_path, 'wb') as f:
                pickle.dump(data_snapshot, f)
            
            # logger.info("💾 [DB] Metadata saved (Async-ish)")
            return True
        except Exception as e:
            logger.error(f"❌ [DB SAVE ERROR] {e}")
            return False