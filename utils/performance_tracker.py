import time
import logging
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Track FPS và performance metrics theo status (pending/identified/confirmed)
    để tìm bottleneck khi confirmed
    """
    def __init__(self, window_size=30, report_interval=10):
        self.window_size = window_size
        self.report_interval = report_interval  # Báo cáo mỗi N giây
        
        # FPS tracking theo status
        self.fps_by_status = {
            'pending': deque(maxlen=window_size),
            'identified': deque(maxlen=window_size),
            'confirmed': deque(maxlen=window_size),
            'overall': deque(maxlen=window_size)
        }
        
        # Frame time tracking
        self.frame_times = {
            'pending': deque(maxlen=window_size),
            'identified': deque(maxlen=window_size),
            'confirmed': deque(maxlen=window_size)
        }
        
        # Component timing
        self.component_times = defaultdict(lambda: deque(maxlen=window_size))
        
        # Queue stats
        self.queue_stats = {
            'ai_queue':deque(maxlen=window_size),
            'cccd_queue': deque(maxlen=window_size),
            'result_queue': deque(maxlen=window_size)
        }
        
        # Counters
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.last_report_time = time.time()
        
        # Status breakdown
        self.status_counts = defaultdict(int)
        self.total_tracks = 0
        
    def record_frame(self, frame_duration, track_manager, ai_queue_size=0, cccd_queue_size=0, result_queue_size=0):
        """
        Ghi nhận frame mới với thông tin chi tiết
        
        Args:
            frame_duration: Thời gian xử lý frame (ms)
            track_manager: TrackManager instance để lấy status
            ai_queue_size: Kích thước AI queue
            cccd_queue_size: Kích thước CCCD queue
            result_queue_size: Kích thước result queue
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Calculate FPS
        time_since_last = current_time - self.last_frame_time
        if time_since_last > 0:
            current_fps = 1.0 / time_since_last
            self.fps_by_status['overall'].append(current_fps)
        
        self.last_frame_time = current_time
        
        # Record queue sizes
        self.queue_stats['ai_queue'].append(ai_queue_size)
        self.queue_stats['cccd_queue'].append(cccd_queue_size)
        self.queue_stats['result_queue'].append(result_queue_size)
        
        # Analyze tracks by status
        status_breakdown = {'pending': 0, 'identified': 0, 'confirmed': 0}
        
        for track_id, obj_data in track_manager.tracked_objects.items():
            status = obj_data.get('status', 'pending')
            if status in status_breakdown:
                status_breakdown[status] += 1
        
        self.total_tracks = len(track_manager.tracked_objects)
        
        # Record FPS by status (ước lượng dựa trên tỷ lệ)
        for status, count in status_breakdown.items():
            self.status_counts[status] = count
            if count > 0 and time_since_last > 0:
                # Ước lượng FPS cho từng status
                estimated_fps = 1.0 / (time_since_last * (count / max(self.total_tracks, 1)))
                self.fps_by_status[status].append(estimated_fps)
                self.frame_times[status].append(frame_duration)
        
    def record_component(self, component_name, duration_ms):
        """Record thời gian xử lý của một component"""
        self.component_times[component_name].append(duration_ms)
    
    def should_report(self):
        """Kiểm tra xem đã đến lúc báo cáo chưa"""
        return (time.time() - self.last_report_time) >= self.report_interval
    
    def get_report(self):
        """Tạo báo cáo chi tiết"""
        current_time = time.time()
        
        # Calculate averages
        report = {
            'timestamp': current_time,
            'frame_count': self.frame_count,
            'total_tracks': self.total_tracks
        }
        
        # FPS by status
        report['fps'] = {}
        for status, fps_list in self.fps_by_status.items():
            if len(fps_list) > 0:
                report['fps'][status] = {
                    'avg': np.mean(fps_list),
                    'min': np.min(fps_list),
                    'max': np.max(fps_list),
                    'std': np.std(fps_list)
                }
            else:
                report['fps'][status] = {'avg': 0, 'min': 0, 'max': 0, 'std': 0}
        
        # Frame times by status
        report['frame_times'] = {}
        for status, times in self.frame_times.items():
            if len(times) > 0:
                report['frame_times'][status] = {
                    'avg': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }
        
        # Queue stats
        report['queues'] = {}
        for queue_name, sizes in self.queue_stats.items():
            if len(sizes) > 0:
                report['queues'][queue_name] = {
                    'avg': np.mean(sizes),
                    'max': np.max(sizes)
                }
        
        # Component times
        report['components'] = {}
        for comp_name, times in self.component_times.items():
            if len(times) > 0:
                report['components'][comp_name] = {
                    'avg': np.mean(times),
                    'max': np.max(times)
                }
        
        # Status counts
        report['status_counts'] = dict(self.status_counts)
        
        return report
    
    def print_report(self):
        """In báo cáo chi tiết ra console"""
        report = self.get_report()
        
        print("\n" + "="*80)
        print(f"📊 PERFORMANCE REPORT - Frame #{self.frame_count}")
        print("="*80)
        
        # Track counts
        print(f"\n🎯 TRACKS: {report['total_tracks']} total")
        for status, count in report['status_counts'].items():
            percentage = (count / max(report['total_tracks'], 1)) * 100
            print(f"   - {status}: {count} ({percentage:.1f}%)")
        
        # FPS by status
        print("\n⚡ FPS BY STATUS:")
        for status in ['pending', 'identified', 'confirmed', 'overall']:
            fps_data = report['fps'].get(status, {})
            if fps_data.get('avg', 0) > 0:
                print(
                    f"   {status:12s}: "
                    f"Avg={fps_data['avg']:5.1f} | "
                    f"Min={fps_data['min']:5.1f} | "
                    f"Max={fps_data['max']:5.1f} | "
                    f"Std={fps_data['std']:4.1f}"
                )
        
        # Frame processing times
        if report.get('frame_times'):
            print("\n⏱️  FRAME PROCESSING TIME (ms):")
            for status, times in report['frame_times'].items():
                print(
                    f"   {status:12s}: "
                    f"Avg={times['avg']:6.1f} | "
                    f"Min={times['min']:6.1f} | "
                    f"Max={times['max']:6.1f}"
                )
        
        # Queue analysis
        if report.get('queues'):
            print("\n📦 QUEUE STATUS:")
            for queue_name, stats in report['queues'].items():
                status_icon = "🔴" if stats['avg'] > 5 else "🟢"
                print(
                    f"   {status_icon} {queue_name:15s}: "
                    f"Avg={stats['avg']:4.1f} | Max={stats['max']:3.0f}"
                )
        
        # Component bottlenecks
        if report.get('components'):
            print("\n🔍 COMPONENT TIMING (Top 5 slowest):")
            sorted_comps = sorted(
                report['components'].items(),
                key=lambda x: x[1]['avg'],
                reverse=True
            )[:5]
            
            for comp_name, times in sorted_comps:
                icon = "🔴" if times['avg'] > 50 else "🟠" if times['avg'] > 20 else "🟢"
                print(
                    f"   {icon} {comp_name:25s}: "
                    f"Avg={times['avg']:6.1f}ms | Max={times['max']:6.1f}ms"
                )
        
        # Bottleneck detection
        self._detect_bottlenecks(report)
        
        print("="*80 + "\n")
        
        # Reset timer
        self.last_report_time = time.time()
    
    def _detect_bottlenecks(self, report):
        """Phát hiện bottleneck tự động"""
        print("\n🚨 BOTTLENECK DETECTION:")
        
        issues = []
        
        # Check FPS drop for confirmed
        confirmed_fps = report['fps'].get('confirmed', {}).get('avg', 0)
        overall_fps = report['fps'].get('overall', {}).get('avg', 0)
        
        if confirmed_fps > 0 and overall_fps > 0:
            fps_drop = ((overall_fps - confirmed_fps) / overall_fps) * 100
            if fps_drop > 30:
                issues.append(
                    f"❌ CONFIRMED FPS DROP: {fps_drop:.0f}% slower than overall "
                    f"({confirmed_fps:.1f} vs {overall_fps:.1f})"
                )
        
        # Check queue buildup
        for queue_name, stats in report.get('queues', {}).items():
            if stats['avg'] > 5:
                issues.append(
                    f"⚠️  {queue_name.upper()} BACKLOG: Avg queue size = {stats['avg']:.1f}"
                )
        
        # Check slow components
        for comp_name, times in report.get('components', {}).items():
            if times['avg'] > 100:
                issues.append(
                    f"🐌 SLOW COMPONENT: {comp_name} taking {times['avg']:.0f}ms"
                )
        
        if issues:
            for issue in issues:
                print(f"   {issue}")
        else:
            print("   ✅ No major bottlenecks detected")
    
    def reset(self):
        """Reset tất cả counters"""
        for status_list in self.fps_by_status.values():
            status_list.clear()
        for times_list in self.frame_times.values():
            times_list.clear()
        for comp_list in self.component_times.values():
            comp_list.clear()
        for queue_list in self.queue_stats.values():
            queue_list.clear()
        
        self.frame_count = 0
        self.status_counts.clear()
