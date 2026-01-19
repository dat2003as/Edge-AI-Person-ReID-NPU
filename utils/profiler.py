import time
import psutil
import os
import numpy as np
from collections import deque

class ResourceProfiler:
    def __init__(self, window_size=30):
        self.history = {} 
        self.window_size = window_size
        self.process = psutil.Process(os.getpid())
        self._start_time = {}
        self._start_ram = {}
        
        # Biến cache để lưu số liệu lần đo cuối
        self.last_cpu_percent = 0.0
        self.last_mem_mb = 0.0

    def start(self, name):
        """Bắt đầu bấm giờ cho tác vụ"""
        self._start_ram[name] = self.process.memory_info().rss 
        self._start_time[name] = time.perf_counter()

    def stop(self, name):
        """Kết thúc bấm giờ"""
        if name in self._start_time:
            elapsed_ms = (time.perf_counter() - self._start_time[name]) * 1000
            current_ram = self.process.memory_info().rss
            ram_diff_bytes = current_ram - self._start_ram.get(name, current_ram)
            ram_diff_mb = max(0, ram_diff_bytes / 1024 / 1024)

            if name not in self.history:
                self.history[name] = {'time': deque(maxlen=self.window_size), 
                                      'ram': deque(maxlen=self.window_size)}
            
            self.history[name]['time'].append(elapsed_ms)
            self.history[name]['ram'].append(ram_diff_mb)

    def get_stats(self):
        """Tính toán thống kê để vẽ lên màn hình"""
        stats = {}
        for name, data in self.history.items():
            if len(data['time']) > 0:
                stats[name] = np.mean(data['time'])
            else:
                stats[name] = 0.0
        
        # --- SỬA ĐỔI ĐỂ CPU LUÔN DƯỚI 100% ---
        # 1. Lấy % CPU thô (có thể > 100% nếu đa nhân)
        raw_cpu = self.process.cpu_percent(interval=None)
        
        # 2. Lấy số lượng nhân CPU (Logical cores)
        num_cores = psutil.cpu_count()
        
        # 3. Chia để quy về thang 0-100% (Tính % của toàn hệ thống)
        # Ví dụ: Máy 8 nhân, app chạy 200% (2 nhân full) -> Hiển thị 25%
        if num_cores and num_cores > 0:
            self.last_cpu_percent = raw_cpu / num_cores
        else:
            self.last_cpu_percent = raw_cpu

        mem_info = self.process.memory_info()
        self.last_mem_mb = mem_info.rss / 1024 / 1024 
        return stats, self.last_cpu_percent, self.last_mem_mb
    
    def print_report(self):
        """
        In báo cáo tổng hợp.
        Đã thêm YOLO, Tracking và Read Camera vào danh sách hiển thị.
        """
        # --- DANH SÁCH CÁC TASK MUỐN HIỂN THỊ ---
        # Tên ở đây phải khớp chính xác với tên trong profiler.start("TÊN")
        SHOW_LIST = [
            "Total_Frame",      # Tổng thời gian 1 khung hình
            "1. Read_Cam",      # Đọc Camera
            "2. YOLO_Track",    # Chạy YOLO
            "3. Gender_Model",     # Phân tích Giới tính
            "3a. Pre-Detection", # Tiền xử lý trước phát hiện
            "3b. Track_Logic", # Logic theo dõi
            "3c. Process_Results", # Xử lý kết quả theo dõi
            "3d. Draw"  ,          # Vẽ lên khung hình 
            "4. Cloth_Color",      # Phân tích Màu áo
            "5. AgeRace_ONNX",     # Phân tích Tuổi/Sắc tộc
            "6. Emotion_Model",    # Phân tích Cảm xúc
            "7. Age_GGNet",      # Phân tích Tuổi (GGNet)
            "Face_MobileFaceNet", # Nếu dùng MobileFaceNet
            "Pose_MediaPipe",
            "Face_Detector (NPU)",
            "web_Update"
        ]

        print(f"\n⚡ --- SYSTEM & MODEL MONITOR [CPU: {self.last_cpu_percent:.1f}% | RAM: {self.last_mem_mb:.0f}MB] ---")
        print(f"{'Component / Task':<25} | {'Latency':<10} | {'FPS (Max)':<10} | {'RAM (MB)':<10} | {'Status'}")
        print("-" * 85)
        
        found_any = False
        for name in SHOW_LIST:
            # Chỉ in nếu task đó đã từng chạy
            if name in self.history and len(self.history[name]['time']) > 0:
                found_any = True
                data = self.history[name]
                avg_time = np.mean(data['time'])
                avg_ram = np.mean(data['ram'])
                
                # Tính FPS lý thuyết (Max FPS nếu chỉ chạy một mình tác vụ này)
                fps_model = 1000.0 / (avg_time + 1e-5)
                
                # Đánh giá màu sắc dựa trên độ trễ
                if avg_time > 100: cost = "🔴 SLOW"
                elif avg_time > 30: cost = "🟠 MED"
                else: cost = "🟢 FAST"

                print(f"{name:<25} | {avg_time:.1f} ms   | {fps_model:.1f}       | {avg_ram:.2f}       | {cost}")
        
        if not found_any:
            print("(Đang chờ dữ liệu... Hệ thống chưa ghi nhận lần chạy nào)")
            
        print("=" * 85 + "\n")