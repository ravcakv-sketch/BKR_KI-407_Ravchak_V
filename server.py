ДОДАТОК А: ЛІСТИНГ КОДУ
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сервер керування наземним роботом з вбудованим вебсервером.

Призначення програми:
- запуск вебсерверу на бортовому комп'ютері Raspberry Pi 4;
- приймання команд оператора через WebSocket;
- керування лівим і правим приводом через VESC-сумісні контролери;
- передавання відеопотоків з передньої CSI-камери та задньої USB-камери;
- перемикання режимів швидкості руху;
- відображення службової інформації про стан системи та напругу батареї;
- реалізація watchdog-механізму для безпечної зупинки при втраті команд.
"""

import asyncio
import glob
import json
import re
import struct
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import cv2
import numpy as np
import serial
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from picamera2 import Picamera2

try:
    # Поворот CSI-камери на рівні libcamera без додаткового навантаження на CPU.
    from libcamera import Transform
except Exception:
    Transform = None  # fallback: програмний поворот через OpenCV


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 8000

# Послідовний інтерфейс для підключення до лівого VESC.
VESC_PORT = "/dev/serial0"
VESC_BAUD = 115200

# Лівий VESC підключений напряму по UART.
# Правий VESC отримує команду через CAN-forward з лівого VESC.
LEFT_VESC_CAN_ID: Optional[int] = None
RIGHT_VESC_CAN_ID: Optional[int] = 2

# Налаштування напрямку обертання приводів.
LEFT_MOTOR_INVERT = False
RIGHT_MOTOR_INVERT = True

# Програмне обмеження максимального duty cycle.
# Апаратні обмеження струму додатково задаються у VESC Tool.
SERVER_MAX_OUTPUT = 0.20

# Плавність наростання duty cycle за секунду.
DUTY_RAMP_PER_SEC = 0.55

# Частота передавання команд до VESC.
MOTOR_HZ = 20.0

# Граничний час відсутності команд, після якого система виконує зупинку.
WATCHDOG_LIMIT_SEC = 0.50

# Мінімальний інтервал між діагностичними повідомленнями у консолі.
MAX_LOG_RATE_SEC = 0.35

# FRONT CSI camera
FRONT_SIZE = (960, 540)
FRONT_FPS = 10
FRONT_JPEG_QUALITY = 58
FRONT_RGB_TO_BGR_FOR_JPEG = False

# Поворот зображення передньої CSI-камери на 180° за потреби монтажу.
FRONT_CAMERA_ROTATE_180 = True

# REAR USB camera
USB_REAR_CAMERA_DEVICE: Optional[str] = None  # наприклад "/dev/video0"; None — автоматичний пошук
REAR_WIDTH = 640
REAR_HEIGHT = 360
REAR_FPS = 8
REAR_JPEG_QUALITY = 50


# ============================================================
# VESC BINARY PROTOCOL
# ============================================================

COMM_GET_VALUES = 4
COMM_SET_DUTY = 5
COMM_FORWARD_CAN = 34


def _crc16(data: bytes) -> int:
    """CRC-16/XMODEM, який використовує VESC."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def _build_vesc_packet(payload: bytes) -> bytes:
    length = len(payload)
    if length < 256:
        header = bytes([0x02, length])
    else:
        header = bytes([0x03, (length >> 8) & 0xFF, length & 0xFF])
    crc = _crc16(payload)
    return header + payload + bytes([(crc >> 8) & 0xFF, crc & 0xFF, 0x03])


def vesc_pkt_get_values() -> bytes:
    """Запит телеметрії з локального VESC: напруга, струм, температура тощо."""
    return _build_vesc_packet(bytes([COMM_GET_VALUES]))


def vesc_pkt_set_duty(duty: float) -> bytes:
    duty_i = int(duty * 100000.0)
    payload = struct.pack("!Bi", COMM_SET_DUTY, duty_i)
    return _build_vesc_packet(payload)


def vesc_pkt_set_duty_can(can_id: int, duty: float) -> bytes:
    duty_i = int(duty * 100000.0)
    payload = struct.pack("!BBBi", COMM_FORWARD_CAN, can_id, COMM_SET_DUTY, duty_i)
    return _build_vesc_packet(payload)


vesc_serial: Optional[serial.Serial] = None
vesc_lock = threading.Lock()
vesc_connected = False
vesc_last_open_attempt = 0.0
VESC_RECONNECT_BACKOFF_SEC = 1.0


def get_vesc_serial() -> Optional[serial.Serial]:
    """Повертає відкритий UART або пробує перепідключити з backoff."""
    global vesc_serial, vesc_connected, vesc_last_open_attempt

    if vesc_serial is not None and vesc_serial.is_open:
        vesc_connected = True
        return vesc_serial

    now = time.time()
    if now - vesc_last_open_attempt < VESC_RECONNECT_BACKOFF_SEC:
        return None
    vesc_last_open_attempt = now

    try:
        vesc_serial = serial.Serial(
            VESC_PORT,
            VESC_BAUD,
            timeout=0.02,
            write_timeout=0.05,
        )
        try:
            vesc_serial.reset_input_buffer()
            vesc_serial.reset_output_buffer()
        except Exception:
            pass
        vesc_connected = True
        print(f"VESC UART connected: {VESC_PORT} @ {VESC_BAUD}")
        return vesc_serial
    except Exception as exc:
        vesc_serial = None
        vesc_connected = False
        print(f"VESC UART connect error: {exc}")
        return None


def close_vesc_serial() -> None:
    global vesc_serial, vesc_connected
    try:
        if vesc_serial is not None:
            vesc_serial.close()
    except Exception:
        pass
    vesc_serial = None
    vesc_connected = False


def vesc_write_duty_raw(left_duty: float, right_duty: float) -> None:
    """Низькорівнева відправка duty у VESC. Викликати тільки під vesc_lock."""
    global vesc_connected

    ser = get_vesc_serial()
    if ser is None:
        vesc_connected = False
        return

    try:
        left_packet = vesc_pkt_set_duty(left_duty) if LEFT_VESC_CAN_ID is None else vesc_pkt_set_duty_can(LEFT_VESC_CAN_ID, left_duty)
        right_packet = vesc_pkt_set_duty(right_duty) if RIGHT_VESC_CAN_ID is None else vesc_pkt_set_duty_can(RIGHT_VESC_CAN_ID, right_duty)
        ser.write(left_packet)
        ser.write(right_packet)
        vesc_connected = True
    except Exception as exc:
        print(f"VESC write error: {exc}")
        close_vesc_serial()



def _vesc_read_packet(ser: serial.Serial, timeout_sec: float = 0.12) -> Optional[bytes]:
    """Прочитати один VESC-пакет і повернути payload без оболонки.

    Викликати тільки під vesc_lock. Функція коротка по timeout, щоб не створювати
    помітних затримок для motor_worker.
    """
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue

        start = b[0]
        if start == 0x02:
            len_b = ser.read(1)
            if len(len_b) != 1:
                continue
            length = len_b[0]
        elif start == 0x03:
            len_b = ser.read(2)
            if len(len_b) != 2:
                continue
            length = (len_b[0] << 8) | len_b[1]
        else:
            continue

        if length <= 0 or length > 512:
            continue

        payload = ser.read(length)
        crc_b = ser.read(2)
        end_b = ser.read(1)

        if len(payload) != length or len(crc_b) != 2 or len(end_b) != 1:
            continue
        if end_b[0] != 0x03:
            continue

        received_crc = (crc_b[0] << 8) | crc_b[1]
        calculated_crc = _crc16(payload)
        if received_crc != calculated_crc:
            continue

        return payload

    return None


def _vesc_parse_values_payload(payload: bytes) -> Optional[dict[str, Any]]:
    """Розібрати COMM_GET_VALUES.

    Для напруги батареї потрібне поле v_in. У типових прошивках VESC воно
    знаходиться на offset 27..28 у payload, де payload[0] == COMM_GET_VALUES.
    Значення передається як int16 / 10.
    """
    if not payload or payload[0] != COMM_GET_VALUES:
        return None
    if len(payload) < 29:
        return None

    try:
        temp_fet = struct.unpack_from("!h", payload, 1)[0] / 10.0
        temp_motor = struct.unpack_from("!h", payload, 3)[0] / 10.0
        current_motor = struct.unpack_from("!i", payload, 5)[0] / 100.0
        current_in = struct.unpack_from("!i", payload, 9)[0] / 100.0
        duty_now = struct.unpack_from("!h", payload, 21)[0] / 1000.0
        erpm = struct.unpack_from("!i", payload, 23)[0]
        v_in = struct.unpack_from("!h", payload, 27)[0] / 10.0
        fault_code = payload[55] if len(payload) > 55 else None
    except Exception:
        return None

    if not (0.0 < v_in < 100.0):
        return None

    return {
        "battery_voltage": round(v_in, 2),
        "vesc_temp_fet": round(temp_fet, 1),
        "vesc_temp_motor": round(temp_motor, 1),
        "vesc_current_motor": round(current_motor, 2),
        "vesc_current_in": round(current_in, 2),
        "vesc_duty_now": round(duty_now, 3),
        "vesc_erpm": int(erpm),
        "vesc_fault_code": fault_code,
    }


def battery_status_from_voltage(voltage: Optional[float]) -> str:
    """Орієнтовні пороги для 12S батареї 48 В.

    Пороги не замінюють VESC cutoff, але дають оператору нормальну індикацію:
    OK >= 44.0 V, LOW >= 42.0 V, нижче 42.0 V — CRITICAL.
    """
    if voltage is None:
        return "UNKNOWN"
    if voltage >= 44.0:
        return "OK"
    if voltage >= 42.0:
        return "LOW"
    return "CRITICAL"


def vesc_read_local_values() -> Optional[dict[str, Any]]:
    """Запитати телеметрію з VESC, який підключений напряму по UART."""
    global vesc_connected

    ser = get_vesc_serial()
    if ser is None:
        vesc_connected = False
        return None

    try:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        ser.write(vesc_pkt_get_values())
        payload = _vesc_read_packet(ser, timeout_sec=0.12)
        if payload is None:
            return None
        parsed = _vesc_parse_values_payload(payload)
        if parsed is not None:
            vesc_connected = True
        return parsed
    except Exception as exc:
        print(f"VESC telemetry error: {exc}")
        close_vesc_serial()
        return None


def vesc_telemetry_worker_loop() -> None:
    """Окремий повільний цикл телеметрії, щоб не змішувати читання з керуванням моторами."""
    while True:
        values = None
        with vesc_lock:
            values = vesc_read_local_values()

        now = time.time()
        with state_lock:
            if values:
                control_state.update(values)
                bv = values.get("battery_voltage")
                control_state["battery_status"] = battery_status_from_voltage(bv)
                control_state["battery_last_update_age_ms"] = 0
                control_state["battery_last_update_time"] = now
            else:
                last = control_state.get("battery_last_update_time")
                if isinstance(last, (int, float)) and last > 0:
                    control_state["battery_last_update_age_ms"] = int((now - last) * 1000)
                else:
                    control_state["battery_last_update_age_ms"] = None

        time.sleep(0.80)


# ============================================================
# CONTROL STATE / MOTOR WORKER
# ============================================================

state_lock = threading.RLock()
workers_lock = threading.Lock()
workers_started = False
last_print_time = 0.0

control_state: dict[str, Any] = {
    "rx_count": 0,
    "armed": False,
    "speed": 0.0,
    "turn": 0.0,
    "left": 0.0,
    "right": 0.0,
    "actual_left": 0.0,
    "actual_right": 0.0,
    "direction": "STOP",
    "camera_mode": "front",   # front/rear/auto
    "main_camera": "front",   # front/rear
    "speed_mode": "NORMAL",
    "speed_mode_max": 0.10,
    "watchdog": "BOOT",
    "stop_reason": "BOOT",
    "last_command_time": time.time(),
    "last_command_age_ms": 0,
    "vesc_connected": False,

    # Телеметрія з VESC / батарея
    "battery_voltage": None,
    "battery_status": "UNKNOWN",
    "battery_last_update_time": 0.0,
    "battery_last_update_age_ms": None,
    "vesc_temp_fet": None,
    "vesc_temp_motor": None,
    "vesc_current_motor": None,
    "vesc_current_in": None,
    "vesc_duty_now": None,
    "vesc_erpm": None,
    "vesc_fault_code": None,
}


def clamp(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
    try:
        value = float(value)
    except Exception:
        value = 0.0
    return max(min_value, min(max_value, value))


def direction_from_values(speed: float, turn: float) -> str:
    if abs(speed) < 0.05 and abs(turn) < 0.05:
        return "STOP"
    if speed > 0.05 and abs(turn) < 0.15:
        return "FORWARD"
    if speed < -0.05 and abs(turn) < 0.15:
        return "BACKWARD"
    if abs(speed) < 0.15 and turn > 0.05:
        return "TURN RIGHT"
    if abs(speed) < 0.15 and turn < -0.05:
        return "TURN LEFT"
    if speed > 0.05 and turn > 0.05:
        return "FORWARD RIGHT"
    if speed > 0.05 and turn < -0.05:
        return "FORWARD LEFT"
    if speed < -0.05 and turn > 0.05:
        return "BACKWARD RIGHT"
    if speed < -0.05 and turn < -0.05:
        return "BACKWARD LEFT"
    return "MOVING"


def apply_motor_inversion(left: float, right: float) -> tuple[float, float]:
    left = clamp(left, -SERVER_MAX_OUTPUT, SERVER_MAX_OUTPUT)
    right = clamp(right, -SERVER_MAX_OUTPUT, SERVER_MAX_OUTPUT)
    if LEFT_MOTOR_INVERT:
        left = -left
    if RIGHT_MOTOR_INVERT:
        right = -right
    return left, right


def ramp_towards(current: float, target: float, max_delta: float) -> float:
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


def set_safe_stop(reason: str, disarm: bool = True) -> None:
    """Змінює тільки стан. Фізичний STOP відправить motor_worker."""
    with state_lock:
        if disarm:
            control_state["armed"] = False
        control_state["speed"] = 0.0
        control_state["turn"] = 0.0
        control_state["left"] = 0.0
        control_state["right"] = 0.0
        control_state["direction"] = "STOP"
        control_state["watchdog"] = reason
        control_state["stop_reason"] = reason
        control_state["last_command_time"] = time.time()


def get_status_snapshot() -> dict[str, Any]:
    now = time.time()
    with state_lock:
        snapshot = dict(control_state)
        snapshot["last_command_age_ms"] = int((now - control_state["last_command_time"]) * 1000)
        snapshot["vesc_connected"] = vesc_connected
        snapshot["server_max_output"] = SERVER_MAX_OUTPUT
    return snapshot


def motor_worker_loop() -> None:
    """
    Єдине місце, яке регулярно пише у VESC.
    Це прибирає зависання від одночасних write з WebSocket/watchdog/keepalive.
    """
    global last_print_time

    period = 1.0 / MOTOR_HZ
    last_loop_time = time.time()
    actual_left = 0.0
    actual_right = 0.0

    while True:
        loop_start = time.time()
        dt = max(0.001, loop_start - last_loop_time)
        last_loop_time = loop_start

        with state_lock:
            age = loop_start - control_state["last_command_time"]
            timed_out = age > WATCHDOG_LIMIT_SEC

            if timed_out:
                if control_state["armed"] or abs(control_state["left"]) > 0.001 or abs(control_state["right"]) > 0.001:
                    control_state["armed"] = False
                    control_state["speed"] = 0.0
                    control_state["turn"] = 0.0
                    control_state["left"] = 0.0
                    control_state["right"] = 0.0
                    control_state["direction"] = "STOP"
                    control_state["watchdog"] = "TIMEOUT"
                    control_state["stop_reason"] = "WATCHDOG_TIMEOUT"
                else:
                    control_state["watchdog"] = "TIMEOUT"

            armed = bool(control_state["armed"])
            target_left = float(control_state["left"]) if armed else 0.0
            target_right = float(control_state["right"]) if armed else 0.0

        # При DISARM/STOP нуль віддаємо негайно, без плавного спаду.
        # Ramp застосовується тільки для руху в ARMED-стані.
        if not armed:
            actual_left = 0.0
            actual_right = 0.0
        else:
            max_delta = DUTY_RAMP_PER_SEC * dt
            actual_left = ramp_towards(actual_left, target_left, max_delta)
            actual_right = ramp_towards(actual_right, target_right, max_delta)
        out_left, out_right = apply_motor_inversion(actual_left, actual_right)

        with vesc_lock:
            vesc_write_duty_raw(out_left, out_right)

        with state_lock:
            control_state["actual_left"] = out_left
            control_state["actual_right"] = out_right
            control_state["vesc_connected"] = vesc_connected

        if loop_start - last_print_time > MAX_LOG_RATE_SEC:
            last_print_time = loop_start
            st = get_status_snapshot()
            print(
                f"armed={st['armed']} speed={st['speed']:+.3f} turn={st['turn']:+.3f} "
                f"targetL={st['left']:+.3f} targetR={st['right']:+.3f} "
                f"outL={st['actual_left']:+.3f} outR={st['actual_right']:+.3f} "
                f"cam={st['main_camera']} wd={st['watchdog']} vesc={vesc_connected}"
            )

        elapsed = time.time() - loop_start
        time.sleep(max(0.001, period - elapsed))


# ============================================================
# VIDEO STREAMS
# ============================================================

JPEG_ENCODE_LOCK = threading.Lock()

front_camera: Optional[Picamera2] = None
front_camera_lock = threading.Lock()
front_camera_software_rotate_180 = False

rear_camera: Optional[cv2.VideoCapture] = None
rear_camera_device: Optional[str] = None
rear_camera_lock = threading.Lock()


class LatestJpegStream:
    def __init__(self, name: str):
        self.name = name
        self.condition = threading.Condition()
        self.part: Optional[bytes] = None
        self.seq = 0
        self.last_update = 0.0
        self.last_error = "NO_FRAME_YET"
        self.clients = 0

    def add_client(self) -> None:
        with self.condition:
            self.clients += 1
            self.condition.notify_all()

    def remove_client(self) -> None:
        with self.condition:
            self.clients = max(0, self.clients - 1)
            self.condition.notify_all()

    def client_count(self) -> int:
        with self.condition:
            return self.clients

    def update(self, part: Optional[bytes], error: str = "") -> None:
        if part is None:
            return
        with self.condition:
            self.part = part
            self.seq += 1
            self.last_update = time.time()
            self.last_error = error
            self.condition.notify_all()

    def wait_for_part(self, last_seq: int, timeout: float = 1.0) -> tuple[int, Optional[bytes]]:
        with self.condition:
            if self.seq == last_seq:
                self.condition.wait(timeout=timeout)
            return self.seq, self.part


front_jpeg_stream = LatestJpegStream("front")
rear_jpeg_stream = LatestJpegStream("rear")


def make_mjpeg_part(frame: np.ndarray, jpeg_quality: int = 80) -> Optional[bytes]:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    try:
        with JPEG_ENCODE_LOCK:
            ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
    except Exception as exc:
        print(f"JPEG encode error: {exc}")
        return None
    if not ok:
        return None
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


def make_text_frame(text: str, width: int = 640, height: int = 360) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y = 110
    for line in text.split("\n"):
        cv2.putText(frame, line, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        y += 44
    return make_mjpeg_part(frame, jpeg_quality=75) or b""


def safe_set_picamera_controls(camera: Picamera2) -> None:
    frame_duration_us = int(1_000_000 / FRONT_FPS)
    try:
        camera.set_controls({
            "AeEnable": True,
            "AwbEnable": True,
            "FrameDurationLimits": (frame_duration_us, frame_duration_us),
            "Sharpness": 1.2,
            "Contrast": 1.05,
            "Saturation": 1.0,
            "Brightness": 0.0,
        })
        print("FRONT CSI controls applied")
    except Exception as exc:
        print(f"FRONT CSI controls warning: {exc}")


def get_front_camera() -> Picamera2:
    global front_camera, front_camera_software_rotate_180
    with front_camera_lock:
        if front_camera is None:
            print("Starting FRONT CSI camera...")
            cam = Picamera2()
            frame_duration_us = int(1_000_000 / FRONT_FPS)

            # Поворот FRONT CSI на 180°.
            # 1) Основний варіант: libcamera Transform(hflip=1, vflip=1) — стабільніше й легше для CPU.
            # 2) Fallback: cv2.rotate у front_capture_loop, якщо transform не підтримується.
            transform_kwargs = {}
            use_software_rotate = False
            if FRONT_CAMERA_ROTATE_180:
                if Transform is not None:
                    transform_kwargs["transform"] = Transform(hflip=1, vflip=1)
                else:
                    use_software_rotate = True

            try:
                config = cam.create_video_configuration(
                    main={"size": FRONT_SIZE, "format": "RGB888"},
                    controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
                    buffer_count=2,
                    **transform_kwargs,
                )
            except TypeError:
                # Старіша версія Picamera2 може не приймати transform/buffer_count/controls.
                # Тоді запускаємо камеру без transform і повертаємо кадр програмно.
                use_software_rotate = FRONT_CAMERA_ROTATE_180
                try:
                    config = cam.create_video_configuration(
                        main={"size": FRONT_SIZE, "format": "RGB888"},
                        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
                        buffer_count=2,
                    )
                except TypeError:
                    config = cam.create_video_configuration(main={"size": FRONT_SIZE, "format": "RGB888"})

            cam.configure(config)
            safe_set_picamera_controls(cam)
            cam.start()
            time.sleep(0.5)
            front_camera = cam
            front_camera_software_rotate_180 = use_software_rotate
            rotation_mode = "software cv2.rotate" if use_software_rotate else "libcamera Transform"
            if FRONT_CAMERA_ROTATE_180:
                print(f"FRONT CSI rotation: 180 deg via {rotation_mode}")
            print(f"FRONT CSI camera started: {FRONT_SIZE[0]}x{FRONT_SIZE[1]} @ {FRONT_FPS} FPS")
        return front_camera


def reset_front_camera() -> None:
    global front_camera, front_camera_software_rotate_180
    with front_camera_lock:
        if front_camera is not None:
            try:
                front_camera.stop()
            except Exception:
                pass
            try:
                front_camera.close()
            except Exception:
                pass
        front_camera = None
        front_camera_software_rotate_180 = False


def front_capture_loop() -> None:
    idle_since: Optional[float] = None
    while True:
        try:
            if front_jpeg_stream.client_count() <= 0:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since > 3.0:
                    reset_front_camera()
                time.sleep(0.2)
                continue

            idle_since = None
            cam = get_front_camera()
            frame = cam.capture_array()
            if FRONT_CAMERA_ROTATE_180 and front_camera_software_rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if FRONT_RGB_TO_BGR_FOR_JPEG:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            front_jpeg_stream.update(make_mjpeg_part(frame, jpeg_quality=FRONT_JPEG_QUALITY))
            time.sleep(1.0 / FRONT_FPS)
        except Exception as exc:
            print(f"FRONT CSI capture error: {exc}")
            front_jpeg_stream.update(make_text_frame("FRONT CSI CAMERA\nCAPTURE ERROR\nRESTARTING"), error=str(exc))
            reset_front_camera()
            time.sleep(1.0)


def video_sort_key(path: str) -> int:
    match = re.search(r"video(\d+)$", path)
    return int(match.group(1)) if match else 9999


def get_usb_candidates() -> list[str]:
    if USB_REAR_CAMERA_DEVICE:
        return [USB_REAR_CAMERA_DEVICE]
    return sorted(glob.glob("/dev/video*"), key=video_sort_key)


def try_open_usb_camera(path: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REAR_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REAR_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REAR_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(12):
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"REAR USB camera opened: {path}, frame={frame.shape}")
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def get_rear_camera() -> Optional[cv2.VideoCapture]:
    global rear_camera, rear_camera_device
    with rear_camera_lock:
        if rear_camera is not None and rear_camera.isOpened():
            return rear_camera
        print("Searching REAR USB camera...")
        for path in get_usb_candidates():
            cap = try_open_usb_camera(path)
            if cap is not None:
                rear_camera = cap
                rear_camera_device = path
                return rear_camera
        rear_camera = None
        rear_camera_device = None
        print("REAR USB camera not found")
        return None


def reset_rear_camera() -> None:
    global rear_camera, rear_camera_device
    with rear_camera_lock:
        if rear_camera is not None:
            try:
                rear_camera.release()
            except Exception:
                pass
        rear_camera = None
        rear_camera_device = None


def rear_capture_loop() -> None:
    idle_since: Optional[float] = None
    while True:
        try:
            if rear_jpeg_stream.client_count() <= 0:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since > 3.0:
                    reset_rear_camera()
                time.sleep(0.2)
                continue

            idle_since = None
            cam = get_rear_camera()
            if cam is None:
                rear_jpeg_stream.update(make_text_frame("REAR USB CAMERA\nNOT FOUND\nCHECK /dev/videoX"), error="not_found")
                time.sleep(1.0)
                continue

            ok, frame = cam.read()
            if not ok or frame is None:
                rear_jpeg_stream.update(make_text_frame("REAR USB CAMERA\nFRAME ERROR\nREOPENING"), error="frame_error")
                reset_rear_camera()
                time.sleep(0.5)
                continue

            rear_jpeg_stream.update(make_mjpeg_part(frame, jpeg_quality=REAR_JPEG_QUALITY))
            time.sleep(1.0 / REAR_FPS)
        except Exception as exc:
            print(f"REAR USB capture error: {exc}")
            rear_jpeg_stream.update(make_text_frame("REAR USB CAMERA\nCAPTURE ERROR\nRESTARTING"), error=str(exc))
            reset_rear_camera()
            time.sleep(1.0)


def mjpeg_stream_from_latest(stream: LatestJpegStream, fallback_text: str, fps: int):
    stream.add_client()
    last_seq = -1
    delay = 1.0 / max(1, fps)
    try:
        while True:
            seq, part = stream.wait_for_part(last_seq, timeout=1.0)
            if part is None:
                part = make_text_frame(fallback_text)
            if part:
                yield part
            last_seq = seq
            time.sleep(delay)
    finally:
        stream.remove_client()


# ============================================================
# FASTAPI LIFESPAN
# ============================================================


def start_background_workers() -> None:
    global workers_started
    with workers_lock:
        if workers_started:
            return
        threading.Thread(target=motor_worker_loop, daemon=True, name="motor-worker").start()
        threading.Thread(target=vesc_telemetry_worker_loop, daemon=True, name="vesc-telemetry").start()
        threading.Thread(target=front_capture_loop, daemon=True, name="front-capture").start()
        threading.Thread(target=rear_capture_loop, daemon=True, name="rear-capture").start()
        workers_started = True
        print("Background workers started: motor + VESC telemetry + front camera + rear camera")


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_background_workers()
    with vesc_lock:
        get_vesc_serial()
    yield
    set_safe_stop("SERVER_SHUTDOWN", disarm=True)
    with vesc_lock:
        vesc_write_duty_raw(0.0, 0.0)
    reset_front_camera()
    reset_rear_camera()
    close_vesc_serial()


app = FastAPI(lifespan=lifespan)


# ============================================================
# HTML UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ground Robot Operator Panel</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
* { box-sizing: border-box; }

html, body {
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    font-family: Arial, sans-serif;
    background: #0f172a;
    color: #e5e7eb;
}

body {
    display: grid;
    grid-template-rows: 38px minmax(0, 1fr);
}

header {
    background: #020617;
    padding: 4px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #334155;
}

h1 { margin: 0; font-size: 17px; white-space: nowrap; }

.status-row {
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: nowrap;
}

.badge {
    padding: 5px 9px;
    border-radius: 999px;
    font-weight: bold;
    font-size: 12px;
    background: #374151;
    white-space: nowrap;
}

.ok   { background: #065f46; }
.bad  { background: #7f1d1d; }
.warn { background: #92400e; }

.app {
    height: 100%;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: 5px;
    padding: 5px;
}

.left-cockpit {
    min-width: 0;
    min-height: 0;
    display: grid;
    grid-template-rows: minmax(0, 1fr) 132px;
    gap: 5px;
}

.bottom-cockpit {
    min-width: 0;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 300px;
    gap: 5px;
}

.side-cockpit {
    min-width: 0;
    min-height: 0;
    display: grid;
    grid-template-rows: auto auto auto minmax(0, 1fr);
    gap: 5px;
}

.panel {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 9px;
    padding: 6px;
    min-width: 0;
    min-height: 0;
}

.camera-panel { padding: 0; overflow: hidden; position: relative; }

.camera-slot-main,
.camera-slot-secondary {
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
}

.camera-card {
    position: relative;
    background: #020617;
    border: 2px dashed #475569;
    border-radius: 9px;
    overflow: hidden;
    width: 100%;
    height: 100%;
    min-height: 0;
}

.camera-card.active-camera {
    border: 3px solid #22c55e;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.18);
}

.camera-card.rear-focus {
    border: 3px solid #f59e0b;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.18);
}

.camera-label {
    position: absolute;
    top: 8px; left: 8px;
    z-index: 2;
    padding: 5px 8px;
    border-radius: 999px;
    background: rgba(15,23,42,0.88);
    color: #e5e7eb;
    font-size: 11px;
    font-weight: bold;
    border: 1px solid #475569;
}

.fixed-camera-mode {
    position: absolute;
    top: 8px; right: 8px;
    z-index: 30;
    display: flex;
    gap: 5px;
}

.camera-mode-button {
    padding: 5px 7px;
    font-size: 11px;
    border-radius: 7px;
    background: rgba(51,65,85,0.92);
}

.camera-mode-button.active { background: #0f766e; }

.camera-stream {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    background: #020617;
}

.motion-panel {
    display: grid;
    grid-template-rows: 34px minmax(0, 1fr);
    gap: 5px;
}

.direction {
    text-align: center;
    font-size: 23px;
    line-height: 1;
    padding: 4px 8px;
    background: #020617;
    border: 1px solid #475569;
    border-radius: 9px;
    font-weight: bold;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.motion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    min-height: 0;
}

.metric-card {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 5px 6px;
    min-width: 0;
    min-height: 0;
}

.metric-label { color: #94a3b8; font-size: 10px; margin-bottom: 1px; }
.metric-value { font-size: 18px; font-weight: bold; line-height: 1.05; word-break: break-word; }
.metric-value.small { font-size: 15px; }

.control-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
}

button {
    border: none;
    border-radius: 8px;
    padding: 8px 7px;
    font-size: 13px;
    font-weight: bold;
    cursor: pointer;
    color: white;
}

.arm    { background: #166534; }
.disarm { background: #92400e; }
.stop   { background: #991b1b; }
.test   { background: #1d4ed8; }

select, input {
    background: #020617;
    color: #e5e7eb;
    border: 1px solid #475569;
    border-radius: 7px;
    padding: 4px;
    width: 100%;
    font-size: 12px;
}

.settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    font-size: 12px;
}

.check-row { display: flex; align-items: center; gap: 6px; }
.check-row input { width: auto; }

.info-line {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid rgba(55,65,81,0.62);
    font-size: 12px;
}

.info-line:last-child { border-bottom: none; }
.info-name { color: #94a3b8; }
.info-value { font-weight: bold; text-align: right; }

.compact-info {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
}

.axes-compact {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
}

.axis-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 5px 6px;
    font-size: 12px;
}

.axis-name { color: #94a3b8; }
.axis-value { font-weight: bold; }

.system-panel {
    display: grid;
    grid-template-rows: 1fr 1fr;
    gap: 6px;
}

.system-block {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 9px;
    padding: 7px;
    min-height: 0;
}

.system-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    color: #94a3b8;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    text-transform: uppercase;
}

.mode-row,
.power-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}

.mode-card,
.power-card {
    background: #020617;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 7px;
    min-width: 0;
}

.mode-card.active-slow { border-color: #22c55e; box-shadow: 0 0 0 1px rgba(34,197,94,0.20); }
.mode-card.active-normal { border-color: #3b82f6; box-shadow: 0 0 0 1px rgba(59,130,246,0.20); }
.mode-card.active-fast { border-color: #f59e0b; box-shadow: 0 0 0 1px rgba(245,158,11,0.20); }

.mode-label,
.power-label {
    color: #94a3b8;
    font-size: 10px;
    margin-bottom: 2px;
}

.mode-value,
.power-value {
    font-size: 20px;
    font-weight: bold;
    line-height: 1.05;
}

.mode-value.small,
.power-value.small { font-size: 16px; }

.mode-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    margin-top: 6px;
}

.mode-buttons.three-modes {
    grid-template-columns: 1fr 1fr 1fr;
}

.mode-button {
    padding: 6px 7px;
    font-size: 12px;
    background: #1d4ed8;
}

.mode-button.active-slow { background: #166534; }
.mode-button.active-normal { background: #1d4ed8; }
.mode-button.active-fast { background: #92400e; }

.power-subline {
    color: #94a3b8;
    font-size: 10px;
    margin-top: 6px;
}

@media (max-width: 1150px) {
    .app { grid-template-columns: minmax(0,1fr) 310px; }
    .bottom-cockpit { grid-template-columns: minmax(0,1fr) 270px; }
    .direction { font-size: 21px; }
    .metric-value { font-size: 16px; }
}

@media (max-height: 850px) {
    body { grid-template-rows: 34px minmax(0,1fr); }
    h1 { font-size: 15px; }
    .badge { padding: 4px 8px; font-size: 11px; }
    .left-cockpit { grid-template-rows: minmax(0,1fr) 118px; }
    .bottom-cockpit { grid-template-columns: minmax(0,1fr) 285px; }
    .motion-panel { grid-template-rows: 30px minmax(0,1fr); }
    .direction { font-size: 19px; padding: 3px 8px; }
    .metric-card { padding: 4px 5px; }
    .metric-value { font-size: 15px; }
    .metric-label { font-size: 9px; }
}
</style>
</head>

<body>

<header>
    <h1>Ground Robot Operator Panel</h1>
    <div class="status-row">
        <span id="wsStatus"         class="badge bad">WS: OFFLINE</span>
        <span id="gamepadStatus"    class="badge bad">TX12: NOT FOUND</span>
        <span id="vescStatus"       class="badge bad">VESC: --</span>
        <span id="batteryStatus"    class="badge warn">BAT: -- V</span>
        <span id="speedModeStatus"  class="badge ok">MODE: NORMAL</span>
        <span id="cameraModeHeader" class="badge ok">CAM: AUTO</span>
        <span id="mainCameraHeader" class="badge ok">MAIN: FRONT</span>
        <span id="watchdogHeader"   class="badge warn">WATCHDOG: --</span>
        <span id="armStatus"        class="badge warn">DISARMED</span>
    </div>
</header>

<div class="app">
    <div class="left-cockpit">
        <div class="panel camera-panel">
            <div class="fixed-camera-mode">
                <button id="cameraAutoBtn"  class="camera-mode-button active" type="button">AUTO</button>
                <button id="cameraFrontBtn" class="camera-mode-button"        type="button">FRONT</button>
                <button id="cameraRearBtn"  class="camera-mode-button"        type="button">REAR</button>
            </div>
            <div id="mainCameraSlot" class="camera-slot-main">
                <div id="frontCameraCard" class="camera-card active-camera">
                    <div id="frontCameraLabel" class="camera-label">FRONT / MAIN</div>
                    <img id="frontVideo" class="camera-stream" src="/video/front" alt="Front camera">
                </div>
            </div>
        </div>

        <div class="bottom-cockpit">
            <div class="panel camera-panel">
                <div id="secondaryCameraSlot" class="camera-slot-secondary">
                    <div id="rearCameraCard" class="camera-card">
                        <div id="rearCameraLabel" class="camera-label">REAR</div>
                        <img id="rearVideo" class="camera-stream" src="/video/rear" alt="Rear camera">
                    </div>
                </div>
            </div>

            <div class="panel motion-panel">
                <div id="direction" class="direction">STOP</div>
                <div class="motion-grid">
                    <div class="metric-card">
                        <div class="metric-label">Speed</div>
                        <div id="speedValue" class="metric-value">0.00</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Turn</div>
                        <div id="turnValue" class="metric-value">0.00</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Left</div>
                        <div id="leftValue" class="metric-value">0.00</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Right</div>
                        <div id="rightValue" class="metric-value">0.00</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="side-cockpit">
        <div class="panel">
            <div class="control-buttons">
                <button id="armBtn"    class="arm">ARM</button>
                <button id="disarmBtn" class="disarm">DISARM</button>
                <button id="stopBtn"   class="stop">STOP</button>
            </div>
            <div style="margin-top: 6px;">
                <div class="info-line">
                    <span class="info-name">ARM axis</span>
                    <span id="armAxisValue" class="info-value">A4: --</span>
                </div>
                <div class="info-line">
                    <span class="info-name">Camera axis</span>
                    <span id="cameraAxisValue" class="info-value">A5: --</span>
                </div>
                <div class="info-line">
                    <span class="info-name">Reverse</span>
                    <span id="reverseAssistValue" class="info-value">OFF</span>
                </div>
                <div class="info-line">
                    <span class="info-name">VESC UART</span>
                    <span id="vescUartInfo" class="info-value">--</span>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="compact-info">
                <div class="metric-card">
                    <div class="metric-label">RX</div>
                    <div id="rxCount" class="metric-value small">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Watchdog</div>
                    <div id="watchdog" class="metric-value small">OK</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Last cmd</div>
                    <div id="lastAge" class="metric-value small">0 ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Stop reason</div>
                    <div id="stopReason" class="metric-value small">BOOT</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="settings-grid">
                <label>Speed
                    <select id="speedAxis">
                        <option value="0">Axis 0</option>
                        <option value="1">Axis 1</option>
                        <option value="2" selected>Axis 2</option>
                        <option value="3">Axis 3</option>
                        <option value="4">Axis 4</option>
                        <option value="5">Axis 5</option>
                    </select>
                </label>
                <label>Turn
                    <select id="turnAxis">
                        <option value="0" selected>Axis 0</option>
                        <option value="1">Axis 1</option>
                        <option value="2">Axis 2</option>
                        <option value="3">Axis 3</option>
                        <option value="4">Axis 4</option>
                        <option value="5">Axis 5</option>
                    </select>
                </label>
                <label>Deadzone
                    <input id="deadzone" type="number" step="0.01" min="0" max="0.3" value="0.06">
                </label>
                <label>Max
                    <input id="maxOutput" type="number" step="0.01" min="0.02" max="0.20" value="0.12" readonly>
                </label>
                <label class="check-row">
                    <input id="invertSpeed" type="checkbox"> Inv speed
                </label>
                <label class="check-row">
                    <input id="invertTurn" type="checkbox" checked> Inv turn
                </label>
            </div>
        </div>

        <div class="panel system-panel">
            <div class="system-block">
                <div class="system-title">
                    <span>Drive speed mode</span>
                    <span id="speedModeSource">MANUAL</span>
                </div>
                <div class="mode-row">
                    <div id="slowModeCard" class="mode-card active-normal">
                        <div class="mode-label">Current mode</div>
                        <div id="speedModeValue" class="mode-value">NORMAL</div>
                    </div>
                    <div id="speedMaxCard" class="mode-card">
                        <div class="mode-label">Max duty</div>
                        <div id="speedModeMaxValue" class="mode-value small">0.10</div>
                    </div>
                </div>
                <div class="mode-buttons three-modes">
                    <button id="slowModeBtn" class="mode-button" type="button">SLOW</button>
                    <button id="normalModeBtn" class="mode-button active-normal" type="button">NORMAL</button>
                    <button id="fastModeBtn" class="mode-button" type="button">FAST</button>
                </div>
            </div>

            <div class="system-block">
                <div class="system-title">
                    <span>Battery</span>
                    <span id="batteryPanelTelemetry">NO DATA</span>
                </div>
                <div class="power-row">
                    <div class="power-card">
                        <div class="power-label">Voltage</div>
                        <div id="batteryPanelValue" class="power-value">-- V</div>
                    </div>
                    <div class="power-card">
                        <div class="power-label">State</div>
                        <div id="batteryPanelState" class="power-value small">UNKNOWN</div>
                    </div>
                </div>
                <div class="power-subline">VESC telemetry input voltage</div>
            </div>
        </div>
    </div>
</div>

<script>
let socket = null;
let armed = false;
let lastJoystickSend = 0;
let tx12WasConnected = false;
let speedMode = "NORMAL";
let speedModeMaxDuty = 0.10;
let lastSpeedSwitchState = null;

let cameraMode = "auto";
let mainCamera = "front";
let pendingMainCamera = null;
let pendingMainCameraSince = 0;

// Auto-camera thresholds are based on the raw speed stick/axis after deadzone,
// not on duty output. This keeps auto switching working even when Max duty is 0.04-0.08.
const AUTO_REAR_ON_AXIS    = -0.18;
const AUTO_FRONT_ON_AXIS   = -0.06;
const AUTO_REAR_DELAY_MS   = 250;
const AUTO_FRONT_DELAY_MS  = 450;

const SPEED_MODE_SLOW_MAX   = 0.08;  // перший режим швидкості
const SPEED_MODE_NORMAL_MAX = 0.12;  // другий режим швидкості
const SPEED_MODE_FAST_MAX   = 0.18;  // третій режим швидкості
const SPEED_MODE_ORDER = ["SLOW", "NORMAL", "FAST"];

// Перемикання швидкості прив'язане до CH8, який у браузері визначається як Axis 7.
// Перемикання камер виконується окремим каналом CH6 / Axis 5.
const SPEED_SWITCH_AXIS = 7;          // CH8 / B / SB -> A7
const SPEED_SWITCH_LOW_LIMIT = -0.60;
const SPEED_SWITCH_HIGH_LIMIT = 0.60;
const SPEED_SWITCH_INVERT = true;     // інверсія положень тумблера швидкості

const CAMERA_SWITCH_AXIS        = 5;
const CAMERA_SWITCH_EDGE_LIMIT  = 0.95;
const ARM_SWITCH_AXIS           = 4;
const ARM_SWITCH_THRESHOLD      = 0.5;
const ARM_SWITCH_INVERT         = false;

const wsStatus         = document.getElementById("wsStatus");
const gamepadStatus    = document.getElementById("gamepadStatus");
const armStatus        = document.getElementById("armStatus");
const watchdogHeader   = document.getElementById("watchdogHeader");
const cameraModeHeader = document.getElementById("cameraModeHeader");
const mainCameraHeader = document.getElementById("mainCameraHeader");
const vescStatus       = document.getElementById("vescStatus");
const batteryStatus    = document.getElementById("batteryStatus");
const speedModeStatus  = document.getElementById("speedModeStatus");
const batteryPanelValue = document.getElementById("batteryPanelValue");
const batteryPanelState = document.getElementById("batteryPanelState");
const batteryPanelTelemetry = document.getElementById("batteryPanelTelemetry");
const speedModeValue = document.getElementById("speedModeValue");
const speedModeMaxValue = document.getElementById("speedModeMaxValue");
const speedModeSource = document.getElementById("speedModeSource");
const slowModeCard = document.getElementById("slowModeCard");
const speedMaxCard = document.getElementById("speedMaxCard");
const slowModeBtn = document.getElementById("slowModeBtn");
const normalModeBtn = document.getElementById("normalModeBtn");
const fastModeBtn = document.getElementById("fastModeBtn");
const vescUartInfo     = document.getElementById("vescUartInfo");

const speedValue   = document.getElementById("speedValue");
const turnValue    = document.getElementById("turnValue");
const leftValue    = document.getElementById("leftValue");
const rightValue   = document.getElementById("rightValue");
const directionLabel = document.getElementById("direction");

const rxCount   = document.getElementById("rxCount");
const watchdog  = document.getElementById("watchdog");
const lastAge   = document.getElementById("lastAge");
const stopReason = document.getElementById("stopReason");

const mainCameraSlot      = document.getElementById("mainCameraSlot");
const secondaryCameraSlot = document.getElementById("secondaryCameraSlot");
const frontCameraCard     = document.getElementById("frontCameraCard");
const rearCameraCard      = document.getElementById("rearCameraCard");
const frontCameraLabel    = document.getElementById("frontCameraLabel");
const rearCameraLabel     = document.getElementById("rearCameraLabel");
const reverseAssistValue  = document.getElementById("reverseAssistValue");
const cameraAxisValue     = document.getElementById("cameraAxisValue");
const armAxisValue        = document.getElementById("armAxisValue");
const cameraAutoBtn       = document.getElementById("cameraAutoBtn");
const cameraFrontBtn      = document.getElementById("cameraFrontBtn");
const cameraRearBtn       = document.getElementById("cameraRearBtn");

function clamp(v, min=-1, max=1) { return Math.max(min, Math.min(max, v)); }

function applyDeadzone(v, dz) {
    return Math.abs(v) < dz ? 0 : v;
}

function getSpeedModeMaxDuty(mode) {
    if (mode === "FAST") return SPEED_MODE_FAST_MAX;
    if (mode === "SLOW") return SPEED_MODE_SLOW_MAX;
    return SPEED_MODE_NORMAL_MAX;
}

function getSpeedModeCssClass(mode) {
    if (mode === "FAST") return "active-fast";
    if (mode === "SLOW") return "active-slow";
    return "active-normal";
}

function setSpeedMode(mode, source = "MANUAL") {
    if (!SPEED_MODE_ORDER.includes(mode)) mode = "NORMAL";
    speedMode = mode;
    speedModeMaxDuty = getSpeedModeMaxDuty(speedMode);

    speedModeValue.textContent = speedMode;
    speedModeMaxValue.textContent = speedModeMaxDuty.toFixed(2);
    speedModeSource.textContent = source;
    speedModeStatus.textContent = `MODE: ${speedMode}`;
    speedModeStatus.className = speedMode === "FAST" ? "badge warn" : "badge ok";

    slowModeCard.classList.remove("active-slow", "active-normal", "active-fast");
    speedMaxCard.classList.remove("active-slow", "active-normal", "active-fast");
    slowModeBtn.classList.remove("active-slow", "active-normal", "active-fast");
    normalModeBtn.classList.remove("active-slow", "active-normal", "active-fast");
    fastModeBtn.classList.remove("active-slow", "active-normal", "active-fast");

    const cls = getSpeedModeCssClass(speedMode);
    slowModeCard.classList.add(cls);
    speedMaxCard.classList.add(cls);
    if (speedMode === "SLOW") slowModeBtn.classList.add(cls);
    else if (speedMode === "FAST") fastModeBtn.classList.add(cls);
    else normalModeBtn.classList.add(cls);

    const maxInput = document.getElementById("maxOutput");
    if (maxInput) maxInput.value = speedModeMaxDuty.toFixed(2);
}

function updateSpeedModeFromGamepad(gp) {
    if (!gp) return;

    const raw = gp.axes[SPEED_SWITCH_AXIS];
    if (raw === undefined || raw === null || Number.isNaN(raw)) {
        speedModeSource.textContent = `B/CH8 A${SPEED_SWITCH_AXIS}: --`;
        return;
    }

    const v = SPEED_SWITCH_INVERT ? -clamp(raw) : clamp(raw);
    const sourceText = `B/CH8 A${SPEED_SWITCH_AXIS}: ${raw.toFixed(3)}`;

    if (v <= SPEED_SWITCH_LOW_LIMIT) {
        if (speedMode !== "SLOW") setSpeedMode("SLOW", sourceText);
        else speedModeSource.textContent = sourceText;
    } else if (v >= SPEED_SWITCH_HIGH_LIMIT) {
        if (speedMode !== "FAST") setSpeedMode("FAST", sourceText);
        else speedModeSource.textContent = sourceText;
    } else {
        if (speedMode !== "NORMAL") setSpeedMode("NORMAL", sourceText);
        else speedModeSource.textContent = sourceText;
    }
}

function directionFromValues(speed, turn) {
    if (Math.abs(speed) < 0.05 && Math.abs(turn) < 0.05) return "STOP";
    if (speed >  0.05 && Math.abs(turn) < 0.15) return "FORWARD";
    if (speed < -0.05 && Math.abs(turn) < 0.15) return "BACKWARD";
    if (Math.abs(speed) < 0.15 && turn >  0.05) return "TURN RIGHT";
    if (Math.abs(speed) < 0.15 && turn < -0.05) return "TURN LEFT";
    if (speed >  0.05 && turn >  0.05) return "FORWARD RIGHT";
    if (speed >  0.05 && turn < -0.05) return "FORWARD LEFT";
    if (speed < -0.05 && turn >  0.05) return "BACKWARD RIGHT";
    if (speed < -0.05 && turn < -0.05) return "BACKWARD LEFT";
    return "MOVING";
}

function setArmedState(nextArmed) {
    armed = Boolean(nextArmed);
    armStatus.textContent = armed ? "ARMED" : "DISARMED";
    armStatus.className   = armed ? "badge ok" : "badge warn";
}

function updateArmFromAxis(axisValue) {
    armAxisValue.textContent = `A4: ${axisValue.toFixed(3)}`;
    let sw = axisValue > ARM_SWITCH_THRESHOLD;
    if (ARM_SWITCH_INVERT) sw = !sw;
    setArmedState(sw);
}

function setCameraMode(mode) {
    cameraMode = mode;
    cameraAutoBtn.classList.remove("active");
    cameraFrontBtn.classList.remove("active");
    cameraRearBtn.classList.remove("active");
    if (mode === "auto") {
        cameraAutoBtn.classList.add("active");
        cameraModeHeader.textContent = "CAM: AUTO";
    } else if (mode === "front") {
        cameraFrontBtn.classList.add("active");
        cameraModeHeader.textContent = "CAM: FRONT";
    } else {
        cameraRearBtn.classList.add("active");
        cameraModeHeader.textContent = "CAM: REAR";
    }
}

function resetPendingCameraSwitch() { pendingMainCamera = null; pendingMainCameraSince = 0; }

function requestMainCamera(camera, delayMs) {
    const now = Date.now();
    if (mainCamera === camera) { resetPendingCameraSwitch(); return; }
    if (pendingMainCamera !== camera) { pendingMainCamera = camera; pendingMainCameraSince = now; return; }
    if (now - pendingMainCameraSince >= delayMs) { setMainCamera(camera); resetPendingCameraSwitch(); }
}

function setMainCamera(camera) {
    const changed = mainCamera !== camera;
    mainCamera = camera;
    frontCameraCard.classList.remove("active-camera", "rear-focus");
    rearCameraCard.classList.remove("active-camera", "rear-focus");
    if (camera === "front") {
        mainCameraSlot.appendChild(frontCameraCard);
        secondaryCameraSlot.appendChild(rearCameraCard);
        frontCameraCard.classList.add("active-camera");
        frontCameraLabel.textContent = "FRONT / MAIN";
        rearCameraLabel.textContent  = "REAR";
        mainCameraHeader.textContent = "MAIN: FRONT";
        reverseAssistValue.textContent = "OFF";
    } else {
        mainCameraSlot.appendChild(rearCameraCard);
        secondaryCameraSlot.appendChild(frontCameraCard);
        rearCameraCard.classList.add("rear-focus");
        frontCameraLabel.textContent = "FRONT";
        rearCameraLabel.textContent  = "REAR / MAIN";
        mainCameraHeader.textContent = "MAIN: REAR";
        reverseAssistValue.textContent = "ON";
    }

    if (changed && socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: "camera", timestamp: Date.now(),
            camera_mode: cameraMode, main_camera: mainCamera,
            speed_mode: speedMode, speed_mode_max: speedModeMaxDuty
        }));
    }
}

function updateCameraFromModeAndSpeed(driveHint) {
    // driveHint: normalized movement direction, approximately -1..+1.
    // Negative = reverse, positive/zero = forward or stop.
    // It is intentionally NOT the limited duty value, because at Max=0.06
    // a threshold like -0.12 would never be reached reliably.
    driveHint = clamp(driveHint);

    if (cameraMode === "front") { setMainCamera("front"); resetPendingCameraSwitch(); return; }
    if (cameraMode === "rear")  { setMainCamera("rear");  resetPendingCameraSwitch(); return; }

    if (driveHint < AUTO_REAR_ON_AXIS) {
        requestMainCamera("rear", AUTO_REAR_DELAY_MS);
    } else if (driveHint > AUTO_FRONT_ON_AXIS) {
        requestMainCamera("front", AUTO_FRONT_DELAY_MS);
    }
}

function updateCameraModeFromAxis(axisValue) {
    cameraAxisValue.textContent = `A5: ${axisValue.toFixed(3)}`;
    if (axisValue <= -CAMERA_SWITCH_EDGE_LIMIT)      setCameraMode("front");
    else if (axisValue >= CAMERA_SWITCH_EDGE_LIMIT)  setCameraMode("rear");
    else                                             setCameraMode("auto");
}

function connectWebSocket() {
    socket = new WebSocket(`ws://${location.host}/ws`);

    socket.onopen = () => {
        wsStatus.textContent = "WS: ONLINE";
        wsStatus.className   = "badge ok";
    };

    socket.onclose = () => {
        wsStatus.textContent = "WS: OFFLINE";
        wsStatus.className   = "badge bad";
        setArmedState(false);
        setTimeout(connectWebSocket, 1000);
    };

    socket.onerror = () => {
        wsStatus.textContent = "WS: ERROR";
        wsStatus.className   = "badge bad";
        setArmedState(false);
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        rxCount.textContent   = data.rx_count ?? 0;
        watchdog.textContent  = data.watchdog ?? "OK";
        watchdogHeader.textContent = `WATCHDOG: ${data.watchdog ?? "OK"}`;
        lastAge.textContent   = `${data.last_command_age_ms ?? 0} ms`;
        stopReason.textContent = data.stop_reason ?? "-";

        watchdogHeader.className = data.watchdog === "OK" ? "badge ok" : "badge warn";

        setArmedState(data.armed);

        // VESC статус
        const vc = data.vesc_connected;
        vescStatus.textContent  = vc ? "VESC: OK" : "VESC: ERR";
        vescStatus.className    = vc ? "badge ok" : "badge bad";
        vescUartInfo.textContent = vc ? "Connected" : "No UART";

        // Напруга батареї з VESC telemetry
        const bv = data.battery_voltage;
        const bs = data.battery_status ?? "UNKNOWN";
        const bAge = data.battery_last_update_age_ms;
        if (bv !== null && bv !== undefined) {
            const voltageText = `${Number(bv).toFixed(1)} V`;
            const telemetryText = (bAge !== null && bAge !== undefined) ? `${Math.round(bAge / 1000)} s` : "LIVE";

            batteryStatus.textContent = `BAT: ${voltageText}`;
            batteryPanelValue.textContent = voltageText;
            batteryPanelState.textContent = bs;
            batteryPanelTelemetry.textContent = telemetryText;

            if (bs === "OK") {
                batteryStatus.className = "badge ok";
                batteryPanelState.style.color = "#86efac";
            } else if (bs === "LOW") {
                batteryStatus.className = "badge warn";
                batteryPanelState.style.color = "#fbbf24";
            } else if (bs === "CRITICAL") {
                batteryStatus.className = "badge bad";
                batteryPanelState.style.color = "#fca5a5";
            } else {
                batteryStatus.className = "badge warn";
                batteryPanelState.style.color = "#e5e7eb";
            }
        } else {
            batteryStatus.textContent = "BAT: -- V";
            batteryStatus.className = "badge warn";
            batteryPanelValue.textContent = "-- V";
            batteryPanelState.textContent = "UNKNOWN";
            batteryPanelState.style.color = "#e5e7eb";
            batteryPanelTelemetry.textContent = "NO DATA";
        }
    };
}

function getFirstGamepad() {
    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const gp of gamepads) { if (gp) return gp; }
    return null;
}

function renderAxes(gp) {
    // Осі не дублюємо у нижньому блоці. Для ARM/CAM залишені окремі поля справа.
}

function updateCommandOnPage(speed, turn, cameraDriveHint = null) {
    speed = clamp(speed); turn = clamp(turn);
    const left  = clamp(speed - turn);
    const right = clamp(speed + turn);
    speedValue.textContent  = speed.toFixed(2);
    turnValue.textContent   = turn.toFixed(2);
    leftValue.textContent   = left.toFixed(2);
    rightValue.textContent  = right.toFixed(2);
    directionLabel.textContent = directionFromValues(speed, turn);

    let hint = cameraDriveHint;
    if (hint === null || Number.isNaN(hint)) {
        const maxOutUi = Math.max(0.01, parseFloat(document.getElementById("maxOutput").value) || 0.08);
        hint = speed / maxOutUi;
    }
    updateCameraFromModeAndSpeed(hint);
}

function sendDrive(speed, turn, cameraDriveHint = null) {
    speed = clamp(speed); turn = clamp(turn);
    const left      = clamp(speed - turn);
    const right     = clamp(speed + turn);
    const direction = directionFromValues(speed, turn);
    updateCommandOnPage(speed, turn, cameraDriveHint);
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: "drive", timestamp: Date.now(),
            armed, speed, turn, left, right, direction,
            camera_mode: cameraMode, main_camera: mainCamera,
            speed_mode: speedMode, speed_mode_max: speedModeMaxDuty
        }));
    }
}

function sendStop(reason = "manual_stop") {
    setArmedState(false);
    updateCommandOnPage(0, 0);
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: "stop", timestamp: Date.now(), reason,
            armed: false, speed: 0, turn: 0, left: 0, right: 0,
            direction: "STOP", camera_mode: cameraMode, main_camera: mainCamera,
            speed_mode: speedMode, speed_mode_max: speedModeMaxDuty
        }));
    }
}

function sendEmergencyStop(reason = "emergency_stop") {
    setArmedState(false);
    updateCommandOnPage(0, 0);
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: "stop", timestamp: Date.now(), reason,
            armed: false, speed: 0, turn: 0, left: 0, right: 0,
            direction: "STOP", camera_mode: cameraMode, main_camera: mainCamera,
            speed_mode: speedMode, speed_mode_max: speedModeMaxDuty
        }));
    }
}

function readJoystickAndSend() {
    const gp = getFirstGamepad();

    if (!gp) {
        gamepadStatus.textContent = "TX12: NOT FOUND";
        gamepadStatus.className   = "badge bad";
        armAxisValue.textContent = "A4: --";
        if (tx12WasConnected) {
            tx12WasConnected = false;
            sendEmergencyStop("TX12_LOST");
        } else {
            setArmedState(false);
            updateCommandOnPage(0, 0);
        }
        requestAnimationFrame(readJoystickAndSend);
        return;
    }

    tx12WasConnected = true;
    gamepadStatus.textContent = "TX12: OK";
    gamepadStatus.className   = "badge ok";
    renderAxes(gp);
    updateArmFromAxis(gp.axes[ARM_SWITCH_AXIS] ?? -1);
    updateCameraModeFromAxis(gp.axes[CAMERA_SWITCH_AXIS] ?? 0);
    updateSpeedModeFromGamepad(gp);

    const speedAxisIndex = parseInt(document.getElementById("speedAxis").value);
    const turnAxisIndex  = parseInt(document.getElementById("turnAxis").value);
    const invertSpeed    = document.getElementById("invertSpeed").checked;
    const invertTurn     = document.getElementById("invertTurn").checked;
    const dz             = parseFloat(document.getElementById("deadzone").value);
    const maxOutput      = speedModeMaxDuty;

    let speedAxisRaw = gp.axes[speedAxisIndex] ?? 0;
    let turnAxisRaw  = gp.axes[turnAxisIndex]  ?? 0;

    if (invertSpeed) speedAxisRaw = -speedAxisRaw;
    if (invertTurn)  turnAxisRaw  = -turnAxisRaw;

    // Для моторів використовуємо duty = raw_axis * Max.
    // Для автоперемикання камер використовуємо саме raw_axis після deadzone,
    // щоб камера реагувала на рух назад навіть при малому Max duty.
    const cameraDriveHint = clamp(applyDeadzone(speedAxisRaw, dz));
    const speed = clamp(cameraDriveHint * maxOutput);
    const turn  = clamp(applyDeadzone(turnAxisRaw, dz) * maxOutput);

    const now = Date.now();
    if (now - lastJoystickSend > 50) {
        lastJoystickSend = now;
        sendDrive(speed, turn, cameraDriveHint);
    }

    requestAnimationFrame(readJoystickAndSend);
}

document.getElementById("armBtn").addEventListener("click",    () => { setArmedState(true); sendDrive(0, 0); });
document.getElementById("disarmBtn").addEventListener("click", () => sendStop("MANUAL_DISARM"));
document.getElementById("stopBtn").addEventListener("click",   () => sendStop("MANUAL_STOP"));

cameraAutoBtn.addEventListener("click",  () => { setCameraMode("auto");  updateCameraFromModeAndSpeed(0); });
cameraFrontBtn.addEventListener("click", () => { setCameraMode("front"); updateCameraFromModeAndSpeed(0); });
cameraRearBtn.addEventListener("click",  () => { setCameraMode("rear");  updateCameraFromModeAndSpeed(0); });
slowModeBtn.addEventListener("click", () => setSpeedMode("SLOW", "PANEL"));
normalModeBtn.addEventListener("click", () => setSpeedMode("NORMAL", "PANEL"));
fastModeBtn.addEventListener("click", () => setSpeedMode("FAST", "PANEL"));

window.addEventListener("gamepadconnected",    (e) => console.log("Gamepad connected:", e.gamepad.id));
window.addEventListener("gamepaddisconnected", ()  => sendEmergencyStop("TX12_DISCONNECTED"));
window.addEventListener("beforeunload",        ()  => sendEmergencyStop("PAGE_UNLOAD"));
window.addEventListener("pagehide",            ()  => sendEmergencyStop("PAGE_HIDE"));
document.addEventListener("visibilitychange",  ()  => { if (document.hidden) sendEmergencyStop("PAGE_HIDDEN"); });

setArmedState(false);
setSpeedMode("NORMAL", "PANEL");
setCameraMode("auto");
setMainCamera("front");
connectWebSocket();
readJoystickAndSend();
</script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})


@app.get("/video/front")
async def video_front():
    return StreamingResponse(
        mjpeg_stream_from_latest(front_jpeg_stream, "FRONT CSI CAMERA\nWAITING FOR FRAME", FRONT_FPS),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/rear")
async def video_rear():
    return StreamingResponse(
        mjpeg_stream_from_latest(rear_jpeg_stream, "REAR USB CAMERA\nWAITING FOR FRAME", REAR_FPS),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/debug/status")
async def debug_status():
    return get_status_snapshot()


@app.get("/debug/video")
async def debug_video():
    return {
        "front": {
            "clients": front_jpeg_stream.client_count(),
            "seq": front_jpeg_stream.seq,
            "last_update_age_sec": round(time.time() - front_jpeg_stream.last_update, 3) if front_jpeg_stream.last_update else None,
            "last_error": front_jpeg_stream.last_error,
            "size": FRONT_SIZE,
            "fps": FRONT_FPS,
        },
        "rear": {
            "clients": rear_jpeg_stream.client_count(),
            "seq": rear_jpeg_stream.seq,
            "last_update_age_sec": round(time.time() - rear_jpeg_stream.last_update, 3) if rear_jpeg_stream.last_update else None,
            "last_error": rear_jpeg_stream.last_error,
            "device": rear_camera_device,
            "size": [REAR_WIDTH, REAR_HEIGHT],
            "fps": REAR_FPS,
        },
    }


@app.get("/debug/vesc")
async def debug_vesc():
    with vesc_lock:
        ser = get_vesc_serial()
        connected = ser is not None and ser.is_open
    return {
        "connected": connected,
        "port": VESC_PORT,
        "baud": VESC_BAUD,
        "left_can_id": LEFT_VESC_CAN_ID,
        "right_can_id": RIGHT_VESC_CAN_ID,
        "server_max_output": SERVER_MAX_OUTPUT,
        "battery_voltage": control_state.get("battery_voltage"),
        "battery_status": control_state.get("battery_status"),
        "battery_last_update_age_ms": control_state.get("battery_last_update_age_ms"),
        "vesc_temp_fet": control_state.get("vesc_temp_fet"),
        "vesc_current_in": control_state.get("vesc_current_in"),
        "vesc_fault_code": control_state.get("vesc_fault_code"),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.10)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps(get_status_snapshot()))
                continue

            now = time.time()
            try:
                data: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = str(data.get("type", "drive"))

            if msg_type == "camera":
                # Камера не має права змінювати швидкісний режим.
                # C/CH6/A5 керує тільки CAM/MAIN, B/CH8/A7 керує тільки MODE.
                with state_lock:
                    control_state["camera_mode"] = str(data.get("camera_mode", control_state["camera_mode"]))
                    control_state["main_camera"] = str(data.get("main_camera", control_state["main_camera"]))
                await websocket.send_text(json.dumps(get_status_snapshot()))
                continue

            if msg_type == "stop":
                reason = str(data.get("reason", "CLIENT_STOP")) or "CLIENT_STOP"
                set_safe_stop(reason, disarm=True)
                with state_lock:
                    control_state["rx_count"] += 1
                    control_state["camera_mode"] = str(data.get("camera_mode", control_state["camera_mode"]))
                    control_state["main_camera"] = str(data.get("main_camera", control_state["main_camera"]))
                    control_state["speed_mode"] = str(data.get("speed_mode", control_state.get("speed_mode", "SLOW")))
                    control_state["speed_mode_max"] = float(data.get("speed_mode_max", control_state.get("speed_mode_max", 0.07)))
                    control_state["watchdog"] = "STOPPED"
                await websocket.send_text(json.dumps(get_status_snapshot()))
                continue

            requested_armed = bool(data.get("armed", False))
            speed = clamp(data.get("speed", 0.0), -SERVER_MAX_OUTPUT, SERVER_MAX_OUTPUT)
            turn = clamp(data.get("turn", 0.0), -SERVER_MAX_OUTPUT, SERVER_MAX_OUTPUT)

            # Диференціальне керування: left = speed - turn, right = speed + turn.
            left = clamp(speed - turn, -SERVER_MAX_OUTPUT, SERVER_MAX_OUTPUT)
            right = clamp(speed + turn, -SERVER_MAX_OUTPUT, SERVER_MAX_OUTPUT)
            direction = direction_from_values(speed, turn)

            with state_lock:
                control_state["rx_count"] += 1
                control_state["last_command_time"] = now
                control_state["armed"] = requested_armed
                control_state["speed"] = speed
                control_state["turn"] = turn
                control_state["left"] = left if requested_armed else 0.0
                control_state["right"] = right if requested_armed else 0.0
                control_state["direction"] = direction
                control_state["camera_mode"] = str(data.get("camera_mode", control_state["camera_mode"]))
                control_state["main_camera"] = str(data.get("main_camera", control_state["main_camera"]))
                control_state["speed_mode"] = str(data.get("speed_mode", control_state.get("speed_mode", "SLOW")))
                control_state["speed_mode_max"] = float(data.get("speed_mode_max", control_state.get("speed_mode_max", 0.07)))
                control_state["watchdog"] = "OK"
                control_state["stop_reason"] = "-" if requested_armed else "DISARMED"

            await websocket.send_text(json.dumps(get_status_snapshot()))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        set_safe_stop("WS_DISCONNECT", disarm=True)
    except Exception as exc:
        print(f"WebSocket error: {exc}")
        set_safe_stop("WS_ERROR", disarm=True)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

