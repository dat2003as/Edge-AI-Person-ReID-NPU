# config.py
import cv2
import os

#======================================================================================

# <<< PATH CONFIGURATION - CẤU HÌNH ĐƯỜNG DẪN >>>

# Vui lòng đảm bảo các file model nằm trong thư mục "models"

# ======================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
# -----------------------------------------------------------------------------
# CẤU HÌNH CHO VIỆC VẼ (DRAWING CONFIGURATIONS)
# -----------------------------------------------------------------------------

# --- Cấu hình cho Bảng thông tin thuộc tính (Info Panel) ---
INFO_PANEL_BG = (40, 40, 40)         # Màu nền của bảng thông tin (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX      # Font chữ chung
FONT_SCALE_INFO = 0.5                # Kích thước font cho thông tin chi tiết
FONT_THICKNESS = 1                   # Độ dày nét chữ

# --- Màu sắc cho các loại văn bản khác nhau ---
COLOR_INFO_TEXT = (255, 255, 255)    # Màu trắng cho thông tin chung (giới tính, da)
COLOR_CLOTHING_TEXT = (200, 200, 200) # Màu xám nhạt cho thông tin quần áo


# # Đường dẫn tới các file model

PERSON_MODEL_PATH = os.path.join(MODEL_DIR, "yolo11n.onnx")
COLTHING_CLASSFIER_MODEL_PATH = os.path.join(MODEL_DIR, "clothing_classifier.onnx")
COLTHING_CLASSFIER_MODEL_CIX_PATH=os.path.join(MODEL_DIR, "clothing_objectdetect_sim_newww177_mainGUD.cix")

GENDER_FACE_MODEL_PATH = os.path.join(MODEL_DIR, "GDF_038_93.pt") # đang chuyển sáng cix
GENDER_POSE_MODEL_PATH = os.path.join(MODEL_DIR, "GDP_038_91.pt") # đang chuyển sáng cix

GENDER_FACE_CIX_PATH = os.path.join(MODEL_DIR, "GDF_038_93_sim_images_percentile.cix")
GENDER_POSE_CIX_PATH = os.path.join(MODEL_DIR, "GDP_038_91_correct_sim.cix")

EMO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
AGE_RACE_MODEL_CIX_PATH = os.path.join(MODEL_DIR, "age_race_model_new_123.cix")
AGE_RACE_MODEL_ONNX_PATH = os.path.join(MODEL_DIR, "best_model_detectFaceAgeAndRace.onnx")
YUNET_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
AGE_RACE_MODEL_GGNET_ONNX_PATH = os.path.join(MODEL_DIR, "age_googlenet.onnx")
GENDER_CIX_MODEL_PATH = os.path.join(MODEL_DIR, "" )
#Model RKNN
#PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n_rknn_model/models/yolo11n-rk3588.rknn")

# GENDER_FACE_MODEL_PATH = os.path.join(BASE_DIR, "GDF_038_93_rknn_model/models/GDF_038_93-rk3588.rknn")

# GENDER_POSE_MODEL_PATH = os.path.join(BASE_DIR, "GDP_038_91_rknn_model/models/GDP_038_91-rk3588.rknn")



POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker.task")
SKIN_CSV_PATH = os.path.join(MODEL_DIR, "skin_tone.csv")



# Danh sách các file model bắt buộc phải có

REQUIRED_MODEL_PATHS = {

"Person Detector": PERSON_MODEL_PATH,

"Pose Estimator": POSE_MODEL_PATH,

"Gender Face Model": GENDER_FACE_MODEL_PATH,

"Gender Pose Model": GENDER_POSE_MODEL_PATH,

"Skin Tone CSV": SKIN_CSV_PATH

}



# ======================================================================================
# <<< PATH CONFIGURATION - CẤU HÌNH ĐƯỜNG DẪN >>>
# ======================================================================================
YuNet_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
YOLO_MODEL_PATH = "models/yolo11n.onnx"
TRACKER_CONFIG_PATH = "botsort.yaml"
MAX_DISAPPEARED_FRAMES_BEFORE_DELETION = 50
ATTRIBUTE_ANALYSIS_INTERVAL = 5
# config.pyq
CENTERFACE_MODEL_PATH = "models/centerface.cix"

# ======================================================================================
# <<< TRACKER LOGIC CONFIGURATION - CẤU HÌNH LOGIC TRACKER >>>
# ======================================================================================
# --- Cấu hình Thu thập & Nhận dạng Thông minh ---
QUALITY_SCORE_THRESHOLD =25  # <<< MỚI >>> Ngưỡng điểm chất lượng để kích hoạt nhận dạng
HIGH_CONF_FACE_SCORE = 10       # <<< MỚI >>> Điểm cộng thêm khi có khuôn mặt rất rõ nét
MID_CONF_FACE_SCORE = 2.0        # <<< MỚI >>> Điểm cộng thêm khi có khuôn mặt khá rõ nét
BASE_REID_SCORE = 2.0             # <<< MỚI >>> Điểm cơ bản cho mỗi lần thu thập được vector toàn thân

STABLE_IDENTIFICATION_THRESHOLD = 0.7# <<< MỚI >>> Ngưỡng điểm tin cậy để coi là 'identified', dưới ngưỡng này là 'tentative'
FACE_CONFIDENCE_THRESHOLD = 0.7 # Ngưỡng tin cậy của model face để tính điểm
# --- Cấu hình cho việc làm giàu dữ liệu (Data Enrichment) ---
# Số lượng vector mặt tối thiểu một ID nên có trong DB. Nếu ít hơn, hệ thống sẽ cố gắng bổ sung.
MAX_FACE_VECTORS_PER_PROFILE =25 # Số vector mặt tối đa
MAX_REID_VECTORS_PER_PROFILE = 25
# Ngưỡng confidence tối thiểu của một khuôn mặt để được xem xét bổ sung vào DB (nên đặt rất cao).
HIGH_CONFIDENCE_THRESHOLD_FOR_ENRICHMENT = 0.95
# --- Cấu hình chung ---
MAX_DISAPPEARED_FRAMES = 10       # Số frame tối đa cho phép một track biến mất trước khi bị xóa
MOVING_AVERAGE_WINDOW = 15        # Kích thước cửa sổ để lưu trữ các vector tạm thời cho mỗi track
# Ngưỡng diện tích bounding box tối thiểu (pixel)
# Bất kỳ box nào có diện tích nhỏ hơn ngưỡng này sẽ bị bỏ qua.
# Ví dụ: 50*80 = 4000
MIN_BBOX_AREA = 20000
MAX_BBOX_AREA = 250000
# ======================================================================================
# <<< VECTOR DATABASE CONFIGURATION - CẤU HÌNH CSDL VECTOR >>>
# ======================================================================================
# --- Namespaces & Dimensions ---
REID_NAMESPACE = "reid_full_body"
FACE_NAMESPACE = "face_features"
OSNET_VECTOR_DIM = 512
FACE_VECTOR_DIM = 128 # Tùy thuộc vào model face của bạn, MobileFaceNet thường là 128 hoặc 512
# Ngưỡng khác biệt tối thiểu để lưu một vector mới (dựa trên khoảng cách Euclidean)
# Nếu khoảng cách giữa vector mới và vector cuối cùng nhỏ hơn ngưỡng này, nó sẽ bị bỏ qua.
VECTOR_DIFFERENCE_THRESHOLD = 0.2
# --- Cấu hình Tìm kiếm & Bỏ phiếu (Voting) ---
SEARCH_TOP_K = 15                 # Lấy K vector gần nhất từ DB để bỏ phiếu

# Ngưỡng cho NHẬN DẠNG KHUÔN MẶT (Face Recognition)
FACE_DB_SEARCH_SIMILARITY_THRESHOLD = 0.55 # Ngưỡng tương đồng để một vector mặt được tính là hợp lệ
FACE_MIN_VOTES_FOR_MATCH =2            # Số phiếu tối thiểu cần có để xác nhận một match từ mặt
MIN_FRAMES_TO_LOCK_METADATA = 50  # Số khung hình tối thiểu để khóa metadata (giới tính, tuổi, cảm xúc)
# Ngưỡng cho NHẬN DẠNG TOÀN THÂN (Re-ID)
REID_DB_SEARCH_SIMILARITY_THRESHOLD = 0.75 # Ngưỡng tương đồng cho vector toàn thân
REID_MIN_VOTES_FOR_MATCH = 5         # Số phiếu tối thiểu cần có để xác nhận một match từ toàn thân

# 🔥 DYNAMIC THRESHOLD MATCHING - Adaptive voting based on score
# Khi score cao → không cần votes cao
DYNAMIC_MATCH_VERY_HIGH_THRESHOLD = 0.85    # Score >= 0.85 → Match ngay (1 vote)
DYNAMIC_MATCH_HIGH_THRESHOLD = 0.75         # 0.75 <= score < 0.85 → Cần 2 votes
DYNAMIC_MATCH_LOW_THRESHOLD = 0.75          # score < 0.75 → Cần 3 votes
DYNAMIC_MATCH_VERY_HIGH_MIN_VOTES = 1       # Votes cần nếu score rất cao
DYNAMIC_MATCH_HIGH_MIN_VOTES = 2            # Votes cần nếu score cao
DYNAMIC_MATCH_LOW_MIN_VOTES = 3             # Votes cần nếu score thấp

# ======================================================================================
# <<< DRAWING CONFIGURATION - CẤU HÌNH HIỂN THỊ >>>
# ======================================================================================
# --- Fonts ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# --- BGR Colors ---
TEMP_ID_COLOR = (0, 0, 255)             # Đỏ - Trạng thái 'pending'
TENTATIVE_ID_COLOR = (0, 165, 255)      # Cam - Trạng thái 'tentative' <<< MỚI >>>
CONFIRMED_ID_COLOR = (0, 255, 0)        # Xanh lá - Trạng thái 'confirmed' hoặc 'identified'
 
YOLO_WIDTH = 640
YOLO_HEIGHT = 480

# ======================================================================================
# <<< PERFORMANCE OPTIMIZATION - FRAME SKIP CONFIGURATION >>>
# ======================================================================================
# YOLO Tracking Frame Skip - Chi chay YOLO moi N+1 frames (5 FPS @ 30fps camera)
YOLO_SKIP_FRAMES =1  # Skip 5 frames, chi chay frame thu 6 (30fps -> 5fps)

# AI Analysis Intervals (seconds) - Tan suat chay cac model AI
CLOTHING_ANALYSIS_INTERVAL = 0.5  # Clothing analysis moi 500ms (2 lan/giay)
EMOTION_ANALYSIS_INTERVAL = 0.5   # Emotion analysis moi 500ms (2 lan/giay)
FACE_QUALITY_INTERVAL = 0.4       # Face quality check moi 400ms

# Queue Sizes - Giam queue size de tranh tich tu
MAX_ATTRIBUTE_QUEUE_SIZE = 10  # Giam tu 20 -> 10
MAX_CCCD_QUEUE_SIZE = 5        # Giam tu 10 -> 5

# ======================================================================================
# <<< CONFIRMED PERSON RE-MATCHING CONFIGURATION >>>
# ======================================================================================
# Temporal window: Thoi gian (second) cho phep de re-match
TEMPORAL_MATCHING_WINDOW = 5

# Face similarity threshold cho confirmed person (cao hon vi trong scene 3m)
CONFIRMED_FACE_SIMILARITY_THRESHOLD = 0.65

# ReID similarity threshold cho confirmed person (fallback neu khong co face)
# Thap hon vi trong scene nho, ReID du unique
CONFIRMED_REID_SIMILARITY_THRESHOLD = 0.55