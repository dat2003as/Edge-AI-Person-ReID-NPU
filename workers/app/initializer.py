"""
System Initialization Module.

Handles initialization of all system components:
- ZMQ sockets
- Databases
- Models (YOLO, YuNet, etc.)
- Workers
- Camera connections
"""

import queue
import threading
import logging
import asyncio
import zmq
from ultralytics import YOLO

import config
from utils.camera import ZMQCamera
from utils.centerface import YuNetDetector
from vector_database import VectorDatabase_Manager
from core.features import Analyzer
from core.attributes import AttributesManager
from core.tracker import TrackManager
from core.cccd.recognition_new import FastCCCDRecognition, DualStreamManager, LLMSender
from core.frame_controller import FrameSkipController
from core.collectors import FaceCropsCollector
from workers.ai_worker import combined_analysis_worker
from workers.cccd_worker import fast_cccd_worker
from workers.result_worker import result_processing_worker

logger = logging.getLogger(__name__)


def initialize_system():
    """
    Initialize all system components.
    
    Returns:
        dict: Dictionary containing all initialized system components:
            - zmq_context: ZMQ Context
            - pub_socket: ZMQ PUB socket
            - db_manager: Vector database manager
            - cccd_db_manager: CCCD vector database manager
            - frame_skip_controller: Frame skip controller
            - reid_face_analyzer: Face analyzer
            - attributes_manager: Attributes manager
            - queues: Dict of all queues (cccd, ai, result, ui)
            - yolo_model: YOLO tracking model
            - yunet_main: YuNet face detector
            - llm_sender: LLM sender for greetings
            - fast_cccd: Fast CCCD recognition
            - dual_stream_manager: Dual stream manager
            - face_collector: Face crops collector
            - track_manager: Track manager
            - zmq_cam: ZMQ camera instance
            - workers: Dict of worker threads
    """
    
    # 1. Initialize ZMQ PUB socket
    logger.info("Initializing ZMQ PUB socket...")
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    
    pub_socket.setsockopt(zmq.SNDHWM, 10)
    pub_socket.setsockopt(zmq.LINGER, 0)
    pub_socket.setsockopt(zmq.SNDTIMEO, 10)
    
    pub_socket.bind("tcp://*:5557")
    print("ZeroMQ PUB: tcp://*:5557 (NON-BLOCKING)")
    
    # 2. Initialize Databases
    logger.info("Init databases...")
    db_manager = VectorDatabase_Manager(index_dir="faiss_indexes")
    cccd_db_manager = VectorDatabase_Manager(index_dir="faiss_indexes_cccd")
    logger.info("Databases ready\n")
    
    # 3. Frame Skip Controller
    frame_skip_controller = FrameSkipController(
        max_age_ms=config.CLOTHING_ANALYSIS_INTERVAL * 1000
    )
    logger.info("Frame Skip Controller ready\n")
    
    # 4. Analyzers
    reid_face_analyzer = Analyzer()
    attributes_manager = AttributesManager(
        reid_face_analyzer=reid_face_analyzer,
        db_manager=db_manager
    )
    
    # 5. Create Queues
    cccd_task_queue = queue.Queue(maxsize=config.MAX_CCCD_QUEUE_SIZE)
    attribute_task_queue = queue.Queue(maxsize=config.MAX_ATTRIBUTE_QUEUE_SIZE)
    attribute_result_queue = queue.Queue()
    ui_queue = queue.Queue(maxsize=5)
    
    logger.info("Queues created (Standard Queue)\n")
    
    # 6. Load Models
    logger.info("Loading models...")
    asyncio.run(attributes_manager.load_models())
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    
    yunet_main = YuNetDetector(
        model_path=config.YuNet_MODEL_PATH,
        input_size=(640, 640),
        score_threshold=0.6,
        nms_threshold=0.85
    )
    logger.info("Models loaded\n")
    
    # 7. LLM & CCCD
    llm_sender = LLMSender(
        endpoint="http://192.168.1.75:5050/greet",
        timeout=5,
        max_retries=3
    )
    
    mobilefacenet = attributes_manager.reid_face_analyzer.face_model
    fast_cccd = FastCCCDRecognition(
        cccd_db_manager=cccd_db_manager,
        mobilefacenet_model=mobilefacenet,
        top_k=5,
    )
    
    cccd_count = cccd_db_manager.count_total_vectors("face")
    logger.info(f"CCCD ready: {cccd_count} vectors\n")
    
    # 8. Dual Stream Manager
    dual_stream_manager = DualStreamManager(db_manager=db_manager)
    logger.info("Dual Stream ready\n")
    
    # 9. Face Collector
    face_collector = FaceCropsCollector(db_manager=db_manager)
    
    # 10. Track Manager
    last_id = db_manager.get_max_person_id()
    track_manager = TrackManager(
        attributes_manager, 
        db_manager,
        dual_stream_manager=dual_stream_manager,
        llm_sender=llm_sender
    )
    track_manager.next_person_id = last_id + 1
    track_manager.frame_skip_controller = frame_skip_controller
    
    logger.info("Track Manager ready (with LLM integration)\n")
    
    # 11. Start Workers
    reid_quality_stats = {
        'total': 0,
        'accepted': 0,
        'rejected_blur': 0,
        'rejected_dark': 0,
        'rejected_contrast': 0,
        'rejected_small': 0,
        'rejected_overexposed': 0
    }
    
    face_quality_stats = {
        'total': 0,
        'accepted': 0,
        'rejected_blur': 0,
        'rejected_low_conf': 0
    }
    
    result_worker = threading.Thread(
        target=result_processing_worker,
        args=(attribute_result_queue, track_manager, reid_quality_stats, face_quality_stats),
        daemon=True,
        name="Result-Worker"
    )
    result_worker.start()
    logger.info("WORKER 3 (Result) started\n")
    
    cccd_worker = threading.Thread(
        target=fast_cccd_worker,
        args=(cccd_task_queue, fast_cccd, llm_sender, dual_stream_manager, 
              track_manager, face_collector, yunet_main, db_manager),
        daemon=True,
        name="CCCD-Worker"
    )
    cccd_worker.start()
    logger.info("WORKER 2 (CCCD) started\n")
    
    ai_worker = threading.Thread(
        target=combined_analysis_worker,
        args=(attribute_task_queue, attribute_result_queue, attributes_manager, 
            track_manager, face_collector, cccd_task_queue),
        daemon=True,
        name="AI-Worker"
    )
    ai_worker.start()
    logger.info("WORKER 1 (AI) started\n")
    
    # 12. Initialize ZMQ Camera
    ZMQ_SERVER_URL = "tcp://192.168.1.66:5556"
    logger.info(f"Connecting to camera via ZeroMQ: {ZMQ_SERVER_URL}")
    
    zmq_cam = ZMQCamera(server_url=ZMQ_SERVER_URL, max_retries=9999, retry_delay=2.0)
    
    if not zmq_cam.isOpened():
        logger.warning(f"Cannot connect to ZMQ camera initially. Will retry in loop...")
    else:
        logger.info("ZMQ Camera connected successfully!")
    
    # Return all components
    return {
        'zmq_context': context,
        'pub_socket': pub_socket,
        'db_manager': db_manager,
        'cccd_db_manager': cccd_db_manager,
        'frame_skip_controller': frame_skip_controller,
        'reid_face_analyzer': reid_face_analyzer,
        'attributes_manager': attributes_manager,
        'queues': {
            'cccd': cccd_task_queue,
            'ai': attribute_task_queue,
            'result': attribute_result_queue,
            'ui': ui_queue
        },
        'yolo_model': yolo_model,
        'yunet_main': yunet_main,
        'llm_sender': llm_sender,
        'fast_cccd': fast_cccd,
        'dual_stream_manager': dual_stream_manager,
        'face_collector': face_collector,
        'track_manager': track_manager,
        'zmq_cam': zmq_cam,
        'workers': {
            'result': result_worker,
            'cccd': cccd_worker,
            'ai': ai_worker
        },
        'quality_stats': {
            'reid': reid_quality_stats,
            'face': face_quality_stats
        }
    }
