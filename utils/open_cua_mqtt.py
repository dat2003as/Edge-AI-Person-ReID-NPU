import paho.mqtt.client as mqtt
import time

# Cấu hình MQTT giống hệt bên ESP32
MQTT_BROKER   = "test.mosquitto.org"
MQTT_PORT     = 1883
MQTT_TOPIC_GO = "test/servo/open"

# Client ID nên khác nhau mỗi lần chạy (hoặc để random)
CLIENT_ID = "python-door-controller-" + str(int(time.time()))

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Đã kết nối MQTT thành công!")
    else:
        print(f"Kết nối thất bại, mã lỗi: {rc}")

def on_publish(client, userdata, mid):
    print("Đã publish lệnh mở cửa thành công!")

# Tạo client
client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
client.on_connect = on_connect
client.on_publish = on_publish

# Kết nối
print(f"Đang kết nối tới broker: {MQTT_BROKER}:{MQTT_PORT}")
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

# Bắt đầu vòng lặp xử lý
client.loop_start()

def open_door(angle=90, wait_ms=3500):
    """
    Gửi lệnh mở cửa
    - angle: góc servo muốn quay tới (0-180)
    - wait_ms: thời gian giữ ở góc đó rồi quay về (ms)
    """
    payload = f"{angle},{wait_ms}"
    # Hoặc dùng format query string: payload = f"angle={angle}&ms={wait_ms}"
    
    result = client.publish(MQTT_TOPIC_GO, payload, qos=1)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"Đã gửi lệnh: angle={angle}, giữ {wait_ms}ms")
    else:
        print("Gửi thất bại!")

# Ví dụ sử dụng
if __name__ == "__main__":
    time.sleep(1)  # chờ kết nối ổn định
    
    # Cách 1: mở như hàm servoOpen() (60 độ, giữ 2 giây)
    open_door(90, 3000)
    
    # Cách 2: mở góc khác, giữ lâu hơn
    # open_door(90, 3500)
    
    # Giữ chương trình chạy để nhận phản hồi (nếu cần)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Tắt chương trình...")
        client.loop_stop()
        client.disconnect()