#core/attributes/models_handler.py
import os
import psutil
import asyncio
import cv2
import config
from concurrent.futures import ThreadPoolExecutor

# --- Import Utils ---
from utils.detectors.mediapipe_pose import HumanDetection
from utils.gender.gender_cix import GenderClassification
from utils.clothing.pose_new import PoseColorAnalyzer
from utils.clothing.clothing_new_cix import ClothingClassifier
from utils.emotion_detect import EmotionEstimator
from utils.detectors.yunet import YuNetDetector


from utils.age_race.age_race_cix import AgeRaceEstimator


class AttributesModelsHandler:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Models
        self.face_detector = None
        self.human_detector = None
        self.gender_classifier = None
        self.clothing_classifier = None
        self.pose_color_analyzer = None
        self.age_race_estimator = None 
        self.emotion_estimator = None
        self.age_ggnet_estimator = None
        self.models_loaded = False

    def get_current_ram(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 # MB

    async def load_models(self):
        loop = asyncio.get_event_loop()
        
        # --- 1. LOAD Yunet ---
        try:
            print(f"Đang tải Yunet: {config.YuNet_MODEL_PATH}")
            if not os.path.exists(config.YuNet_MODEL_PATH):
                print(f"❌ KHÔNG TÌM THẤY MODELYUNET.")
            else:
                mem_before = self.get_current_ram()
                self.face_detector = YuNetDetector(config.YuNet_MODEL_PATH)
                print("✅ Yunet Loaded.")
                print(f"📦 [RAM] Yunet: {self.get_current_ram() - mem_before:.1f} MB")
        except Exception as e:
            print(f"❌ Lỗi load CenterFace: {e}")
            self.face_detector = None
            

        # --- 2. LOAD CÁC MODEL KHÁC ---
        mem_before = self.get_current_ram()
        self.human_detector = await loop.run_in_executor(
            self.executor, lambda: HumanDetection(config.PERSON_MODEL_PATH, config.POSE_MODEL_PATH)
        )
        print(f"📦 [RAM] Human Detector: {self.get_current_ram() - mem_before:.1f} MB")

        mem_before = self.get_current_ram()
        self.gender_classifier = await loop.run_in_executor(
            self.executor, lambda: GenderClassification(config.GENDER_FACE_CIX_PATH, config.GENDER_POSE_CIX_PATH)
        )
        print(f"📦 [RAM] Gender Model:   {self.get_current_ram() - mem_before:.1f} MB")

        mem_before = self.get_current_ram()
        print("Đang tải Clothing Classifier...")
        self.clothing_classifier = await loop.run_in_executor(
            self.executor, lambda: ClothingClassifier(config.SKIN_CSV_PATH, config.COLTHING_CLASSFIER_MODEL_CIX_PATH)
        )
        print(f"📦 [RAM] Clothing Model: {self.get_current_ram() - mem_before:.1f} MB")

        mem_before = self.get_current_ram()
        self.pose_color_analyzer = PoseColorAnalyzer()
        print(f"📦 [RAM] Pose/Color Analyzer: {self.get_current_ram() - mem_before:.1f} MB")
        
        cix_path = getattr(config, 'AGE_RACE_MODEL_CIX_PATH', None)
        if cix_path and os.path.exists(cix_path):
             if AgeRaceEstimator:
                 try:
                     self.age_race_estimator = await loop.run_in_executor(
                         self.executor, lambda: AgeRaceEstimator(cix_path)
                     )
                     print("Load Age/Race CIX thanh cong!")
                     print(f"[RAM] Age/Race CIX:  {self.get_current_ram() - mem_before:.1f} MB")
                 except Exception as e:
                     print(f"Loi Crash khi load CIX: {e}")
             else:
                 print("Loi: Chua import duoc class AgeRaceEstimator")

        self.emotion_estimator = await loop.run_in_executor(
            self.executor, lambda: EmotionEstimator(config.EMO_MODEL_PATH)
        )   
        print(f"📦 [RAM] Emotion Model:  {self.get_current_ram() - mem_before:.1f} MB")
        
        

        self.models_loaded = True
        print("✅ Attribute Models Loaded Process Finished.\n")

    # --- Wrapper methods for prediction ---
    def predict_gender(self, img, keypoints):
        """
        🔥 FIXED: Thêm log để debug Gender confidence
        """
        result = self.gender_classifier.predict(img, keypoints)
        
        # 🔥 THÊM LOG CHI TIẾT
        print(f"🔍 [GENDER RAW OUTPUT]")
        print(f"   - Type: {type(result)}")
        print(f"   - Content: {result}")
        
        if isinstance(result, dict):
            print(f"   - gender: {result.get('gender', 'N/A')}")
            print(f"   - confidence: {result.get('confidence', 0.0):.4f}")
            print(f"   - strategy: {result.get('strategy', 'N/A')}")
        else:
            print(f"   ⚠️ WARNING: Result is not a dict!")
        
        return result

    def detect_pose(self, frame, bbox):
        return self.human_detector.detect_pose_from_bbox(frame, bbox)

    async def predict_clothing_async(self, image, keypoints, kpts_z, external_data):
        # Gọi qua PoseColorAnalyzer để sinh regional_analysis
        pose_result = await self.pose_color_analyzer.process_and_classify(
            image=image,
            keypoints=keypoints,
            classifier=self.clothing_classifier,
            kpts_z=kpts_z,
            external_data=external_data
        )
        return pose_result['classification']
        
        # if pose_result is None:
        #     logger.warning("[Clothing] PoseColorAnalyzer trả về None → fallback YOLO only")
        #     # Fallback chỉ YOLO nếu pose fail
        #     return self.clothing_classifier.classify({
        #         'image': image,
        #         'keypoints': keypoints,
        #         'kpts_z': kpts_z,
        #         'external_data': external_data,
        #         'regional_analysis': {}  # vẫn rỗng nhưng có thể thêm fallback sau
        #     })
        
        # return pose_result['classification']

    def predict_age_race(self, face_input):
        if self.age_race_estimator:
            return self.age_race_estimator.predict(face_input)
        return None

    def predict_emotion(self, face_gray):
        return self.emotion_estimator.predict(face_gray)

    def predict_age_ggnet(self, face_img):
        if self.age_ggnet_estimator:
            return self.age_ggnet_estimator.predict(face_img)
        return None