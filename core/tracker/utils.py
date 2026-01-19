# core/tracker/utils.py
import numpy as np
import config

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("LỖI: Thư viện scikit-learn chưa được cài đặt.")
    KMeans = None

class TrackerUtils:
    @staticmethod
    def find_dominant_color(colors, k=3):
        if not colors or not KMeans:
            return None
        
        pixels = np.array(colors)
        if len(pixels) < k:
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant = unique_colors[counts.argmax()]
            return dominant.tolist()

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        
        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_label = unique_labels[counts.argmax()]
        
        dominant_color = kmeans.cluster_centers_[dominant_cluster_label]
        return dominant_color.astype(int).tolist()

    @staticmethod
    def get_query_vector(vectors_deque):
        """Tính trung bình vector từ một deque."""
        if not vectors_deque: return None
        return np.mean(np.array(list(vectors_deque)), axis=0).tolist()

    @staticmethod
    def get_query_vector_face(face_vectors_deque):
        """Tính trung bình vector khuôn mặt với ngưỡng tin cậy."""
        valid_vectors = [v for v, c in face_vectors_deque if c >= config.FACE_CONFIDENCE_THRESHOLD]
        if not valid_vectors: return None
        return np.mean(np.array(valid_vectors), axis=0).tolist()