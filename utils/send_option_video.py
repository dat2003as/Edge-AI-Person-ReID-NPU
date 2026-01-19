#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/send_option_video.py
import logging
from urllib import response
import requests
logger = logging.getLogger(__name__)

VIDEO_SERVER_URL = "http://localhost:9999/play-video"

def send_gender_to_server(gender):
    """
    Gửi giới tính đến video server để phát video
    Args:
        gender: 'male' hoặc 'female'
    """
    try:
        response = requests.post(
            VIDEO_SERVER_URL,
            json={'gender': gender},
            timeout=2
        )
        print(f"DEBUG - Status code thực tế: {response.status_code}")
        print(f"DEBUG - Response text: {response.text}")
        logger.info(f"DEBUG - Status: {response.status_code} | Body: {response.text}")
        if response.status_code == 200:
            logger.info(f"✅ [VIDEO] Đã gửi gender={gender} đến server thành công")
            return True
        elif response.status_code == 429:
            logger.debug(f"⏳ [VIDEO] Server đang phát video, đợi...")
            return False
        else:
            logger.warning(f"⚠️ [VIDEO] Server trả về lỗi: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ [VIDEO] Không thể kết nối đến server {VIDEO_SERVER_URL}")
        logger.error(f"💡 Hãy chạy: python video_server.py")
        return False
    except Exception as e:
        logger.error(f"❌ [VIDEO] Lỗi khi gửi request: {e}")
        return False
    
def send_table_to_server(table):
    """
    Gửi bảng đến video server để phát video
    Args:
        table: dữ liệu bảng (ví dụ: '1', '2', '3')
    """
    if not isinstance(table, str):  # Giả sử table là str; chỉnh nếu cần
        logger.error(f"❌ [VIDEO] Table phải là string, nhận: {type(table)}")
        return False
    try:
        response = requests.post(
            VIDEO_SERVER_URL,
            json={'table': table},
            timeout=2
        )
        if response.status_code == 200:
            logger.info(f"✅ [VIDEO] Đã gửi table={table} đến server thành công")
            return True
        elif response.status_code == 429:
            logger.debug(f"⏳ [VIDEO] Server đang phát video, đợi...")
            return False
        else:
            logger.warning(f"⚠️ [VIDEO] Server trả về lỗi: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ [VIDEO] Không thể kết nối đến server {VIDEO_SERVER_URL}")
        logger.error(f"💡 Hãy chạy: python video_server.py")
        return False
    except Exception as e:
        logger.error(f"❌ [VIDEO] Lỗi khi gửi request: {e}")
        return False