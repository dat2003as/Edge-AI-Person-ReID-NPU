import cv2
import zmq
import logging
import time
import platform

# --- Config ---
class CameraConfig:
    WIDTH = 2560
    HEIGHT = 1440
    FPS = 24 # Đẩy lên 30 cho mượt
    ZMQ_PORT = 5556
    USE_GSTREAMER = True  # Đặt True khi chạy trên Orange Pi/RPi
    GSTREAMER_DEVICE = "/dev/video0"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gstreamer_pipeline(device, width, height, fps):
    """
    Tối ưu hóa Pipeline với drop=1 để giảm Latency xuống thấp nhất
    """
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=True max-buffers=1" 
    )

def main_loop():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # Rất tốt! Bạn đã dùng CONFLATE, đây là keyword cho Low Latency ZMQ.
    # Nó giúp Client chỉ nhận message cuối cùng, không nhận message cũ.
    socket.setsockopt(zmq.CONFLATE, 1) 
    
    socket.bind(f"tcp://0.0.0.0:{CameraConfig.ZMQ_PORT}")
    
    cap = None
    frame_count = 0

    try:
        system_os = platform.system()
        using_gstreamer = CameraConfig.USE_GSTREAMER

        if using_gstreamer and system_os == "Linux":
            logger.info("Starting GStreamer Pipeline (Low Latency Mode)...")
            pipeline = get_gstreamer_pipeline(
                CameraConfig.GSTREAMER_DEVICE, 
                CameraConfig.WIDTH, 
                CameraConfig.HEIGHT, 
                CameraConfig.FPS
            )
            logger.info(f"Pipeline: {pipeline}")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            logger.info("Using standard VideoCapture...")
            cap = cv2.VideoCapture(0)
            # Lưu ý: Webcam thường không set được FPS chính xác nếu không hỗ trợ
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CameraConfig.FPS)

        if not cap.isOpened():
            logger.error("Failed to open camera!")
            return

        logger.info(f"Camera Server started on port {CameraConfig.ZMQ_PORT}")

        while True:
            # cap.read() đã tự động chờ (blocking) theo FPS của camera rồi.
            # KHÔNG ĐƯỢC DÙNG time.sleep() Ở ĐÂY.
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Failed to read frame")
                continue

            frame_count += 1
            
            # Mẹo: Giảm quality xuống 80-85 là đủ đẹp, giúp gửi qua mạng nhanh hơn
            ret_code, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            if ret_code:
                socket.send(buffer)

            if frame_count % 100 == 0:
                logger.info(f"Streaming frame {frame_count}...")

    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if cap:
            cap.release()
        socket.close()
        context.term()

if __name__ == "__main__":
    main_loop()