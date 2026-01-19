#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# video_server.py - Web Video Player Server (FastAPI) với Cooldown theo loại video
import logging
import os
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(title="Web Video Player Server with Type Cooldown")

# ====================== CẤU HÌNH ======================
VIDEO_FOLDER = "./video"
VIDEO_MALE = "tom-ford-fougere-d-argent-spec-ad-1440-publer.io.mp4"
VIDEO_FEMALE = "tvc-quang-cao-nuoc-hoa-cao-cap-coco-chanel-1440-publer.io.mp4"
VIDEO_TABLE1 = "tvc-quang-cao-nuoc-hoa-cao-cap-coco-chanel-1440-publer.io.mp4"
VIDEO_TABLE2 = "video_ban2.mp4"
VIDEO_TABLE3 = "video_ban3.mp4"

COOLDOWN_SECONDS = 120  # 2 phút cooldown cho cùng loại gender

# Trạng thái toàn cục
current_video_path: Optional[str] = None
current_label: str = "Waiting..."
video_playing: bool = False

# Cooldown theo loại
last_male_time: float = 0.0
last_female_time: float = 0.0

# Mount thư mục video
app.mount("/videos", StaticFiles(directory=VIDEO_FOLDER), name="videos")


# ====================== MODEL REQUEST ======================
class VideoRequest(BaseModel):
    gender: Optional[str] = None
    table: Optional[str] = None


# ====================== TRANG CHỦ - VIDEO PLAYER WEB ======================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Web Video Player</title>
        <style>
            body { margin:0; padding:0; background:#000; font-family:Arial; color:#fff; overflow:hidden; height:100vh; }
            #container { position:relative; width:100vw; height:100vh; }
            video { width:100%; height:100%; object-fit:contain; background:#111; }
            #overlay { position:absolute; top:20px; left:20px; background:rgba(0,0,0,0.65); padding:15px 20px; border-radius:12px; max-width:420px; font-size:1.2em; box-shadow:0 4px 15px rgba(0,0,0,0.6); }
            #status { font-size:1.6em; margin-bottom:8px; font-weight:bold; }
            #info { color:#00ff88; }
        </style>
    </head>
    <body>
        <div id="container">
            <video id="player" controls muted>
                <source id="videoSource" src="" type="video/mp4">
                Trình duyệt không hỗ trợ phát video.
            </video>
            <div id="overlay">
                <div id="status">⏳ Đang chờ request...</div>
                <div id="info">Current: Waiting...</div>
            </div>
        </div>

        <script>
            const player = document.getElementById('player');
            const source = document.getElementById('videoSource');
            const statusEl = document.getElementById('status');
            const infoEl = document.getElementById('info');
            let currentVideoUrl = '';

            player.addEventListener('ended', () => {
                statusEl.textContent = '✅ Đã phát xong. Chờ request mới...';
                infoEl.textContent = 'Current: Waiting...';
                source.src = '';
                player.load();
                currentVideoUrl = '';
            });

            function updateVideo(videoUrl, label) {
                if (videoUrl === currentVideoUrl) return;
                currentVideoUrl = videoUrl;
                statusEl.textContent = `▶️ Đang phát: ${label}`;
                infoEl.textContent = `Current: ${label}`;
                source.src = videoUrl;
                player.load();
                player.play().catch(err => {
                    console.warn("Auto-play bị chặn:", err);
                    statusEl.textContent = `▶️ ${label} sẵn sàng - nhấn Play nếu cần`;
                });
            }

            async function checkStatus() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    if (data.playing && data.video_url) {
                        updateVideo(data.video_url, data.label);
                    } else {
                        if (currentVideoUrl !== '') {
                            source.src = '';
                            player.load();
                            currentVideoUrl = '';
                        }
                        statusEl.textContent = '⏳ Đang chờ request...';
                        infoEl.textContent = 'Current: Waiting...';
                    }
                } catch (e) {
                    console.error('Lỗi poll status:', e);
                }
            }

            setInterval(checkStatus, 2000);
            checkStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ====================== STATUS ======================
@app.get("/status")
async def get_status():
    global current_video_path, current_label, video_playing

    if current_video_path and os.path.exists(current_video_path):
        video_url = f"/videos/{os.path.basename(current_video_path)}"
        return {
            "playing": video_playing,
            "video_url": video_url,
            "label": current_label
        }
    else:
        video_playing = False
        return {"playing": False, "video_url": None, "label": "Waiting..."}


# ====================== PHÁT VIDEO VỚI COOLDOWN THEO LOẠI ======================
@app.post("/play-video")
async def play_video_api(req: VideoRequest):
    global current_video_path, current_label, video_playing
    global last_male_time, last_female_time

    if video_playing:
        raise HTTPException(
            status_code=429,
            detail="Đang phát video, vui lòng đợi video hiện tại kết thúc"
        )

    current_time = time.time()
    video_path = None
    label = None
    is_male = False

    # Xác định loại video
    if req.table:
        table = str(req.table).strip()
        if table == '1':
            video_path = os.path.join(VIDEO_FOLDER, VIDEO_TABLE1)
            label = "Table 1"
        elif table == '2':
            video_path = os.path.join(VIDEO_FOLDER, VIDEO_TABLE2)
            label = "Table 2"
        elif table == '3':
            video_path = os.path.join(VIDEO_FOLDER, VIDEO_TABLE3)
            label = "Table 3"
        else:
            raise HTTPException(400, detail=f"Table không hợp lệ: {table}")
        # Table luôn được chấp nhận (không cooldown)

    elif req.gender:
        gender = req.gender.lower().strip()
        if gender in ['male', 'nam', 'man']:
            is_male = True
            cooldown_time = last_male_time
            video_path = os.path.join(VIDEO_FOLDER, VIDEO_MALE)
            label = "Nam giới 👨"
        elif gender in ['female', 'nữ', 'nu', 'woman']:
            is_male = False
            cooldown_time = last_female_time
            video_path = os.path.join(VIDEO_FOLDER, VIDEO_FEMALE)
            label = "Nữ giới 👩"
        else:
            raise HTTPException(400, detail=f"Giới tính không hợp lệ: {req.gender}")

        # Kiểm tra cooldown cho gender
        if current_time - cooldown_time < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - (current_time - cooldown_time))
            raise HTTPException(
                status_code=429,
                detail=f"Video loại này đang cooldown, vui lòng đợi {remaining} giây"
            )

    else:
        raise HTTPException(400, detail="Yêu cầu phải cung cấp 'gender' hoặc 'table'")

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(404, detail=f"Video không tồn tại: {video_path}")

    # Cập nhật trạng thái & cooldown
    current_video_path = video_path
    current_label = label
    video_playing = True

    if is_male:
        last_male_time = current_time
    else:
        last_female_time = current_time

    logger.info(f"🎬 Phát video: {label} → {os.path.basename(video_path)} "
                f"(Cooldown cập nhật: {time.strftime('%H:%M:%S', time.localtime(current_time))})")

    return {
        "success": True,
        "message": f"Đang phát 1 lần: {label}",
        "video_url": f"/videos/{os.path.basename(video_path)}",
        "label": label
    }


# ====================== KHỞI ĐỘNG ======================
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Web Video Player Server khởi động với Cooldown theo loại...")
    logger.info(f"📂 Thư mục video: {os.path.abspath(VIDEO_FOLDER)}")
    logger.info("🌐 Truy cập: http://localhost:9999")
    logger.info("📡 API: POST http://localhost:9999/play-video")
    uvicorn.run(app, host="0.0.0.0", port=9999, log_level="info")