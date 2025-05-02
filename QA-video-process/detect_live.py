import sys
import time
import argparse
import cv2
import numpy as np
import torch
import mss

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox
)

# YOLO v7 관련 임포트 (환경에 맞게 수정)
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

# mss를 이용한 화면 캡쳐 설정 (첫 번째 모니터)
sct = mss.mss()
monitor = sct.monitors[1]  # primary monitor

# 각 클래스별 최대 검출 개수 설정
max_items_per_class = {
    "main_Play": 1,
    "inGame_Player": 1,
    "inGame_block": 99,
    "inGame_Danger": 99,
    "main_Build": 1,
    "main_Custom": 1,
    "main_Link": 1,
    "main_Daily": 1,
    "main_Quit": 1,
    "main_Player": 1,
    "main_Status": 1,
    "main_Ranking": 1,
    "main_Option": 1
}
default_max_items = 10


class MainWindow(QMainWindow):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

        self.setWindowTitle("YOLO v7 - Game State & AutoQA")
        self.resize(1400, 800)

        # detection 활성화 여부 (시작/정지)
        self.capture_active = False

        # 상태 및 성능 관련 변수
        self.current_state = "대기중"
        self.last_state = None
        self.last_update_time = time.time()
        self.fps = 0
        self.inference_time = 0

        # 설정 파라미터 (UI에서 조절 가능)
        self.conf_thresh = 0.25
        self.iou_thresh = 0.45
        self.update_interval = 30  # ms

        self.init_ui()

        # 타이머 (업데이트는 시작/정지 버튼에 따라 제어)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(self.update_interval)

    def init_ui(self):
        """메인 UI 구성"""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 왼쪽 영역 : 이미지 디스플레이와 탐지 객체 리스트
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        # 이미지 디스플레이
        self.image_label = QLabel("화면 로딩 중...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(800, 600)
        left_layout.addWidget(self.image_label)

        # 탐지 객체 리스트
        self.detection_list_edit = QTextEdit()
        self.detection_list_edit.setReadOnly(True)
        left_layout.addWidget(self.detection_list_edit, stretch=1)

        # 오른쪽 영역 : 컨트롤, 상태, 성능, 로그, 설정 패널
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)

        # [Control Panel]
        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        # [State Panel]
        state_group = QGroupBox("현재 상태")
        state_layout = QVBoxLayout()
        self.state_label = QLabel("상태: 대기중")
        self.counts_label = QLabel("Main: 0 | InGame: 0 | InPlay: 0")
        state_layout.addWidget(self.state_label)
        state_layout.addWidget(self.counts_label)
        state_group.setLayout(state_layout)
        right_layout.addWidget(state_group)

        # [Performance Panel]
        perf_group = QGroupBox("성능 모니터링")
        perf_layout = QVBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.inf_time_label = QLabel("추론 시간: 0 ms")
        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.inf_time_label)
        perf_group.setLayout(perf_layout)
        right_layout.addWidget(perf_group)

        # [Log Panel]
        log_group = QGroupBox("상태 전환 로그")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        # [Settings Panel]
        settings_group = QGroupBox("설정")
        settings_layout = QFormLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(self.conf_thresh)
        self.conf_spin.valueChanged.connect(self.on_conf_thresh_changed)
        settings_layout.addRow("Confidence Threshold:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(self.iou_thresh)
        self.iou_spin.valueChanged.connect(self.on_iou_thresh_changed)
        settings_layout.addRow("IoU Threshold:", self.iou_spin)

        settings_group.setLayout(settings_layout)
        right_layout.addWidget(settings_group)

        right_layout.addStretch(1)

    def start_detection(self):
        """시작 버튼 클릭 시 탐지 시작 및 기준 시간 기록"""
        if not self.capture_active:
            self.capture_active = True
            self.start_time = time.time()  # 탐지 시작 기준 시간 기록
            self.log_message("탐지 시작")
            self.last_update_time = time.time()
            self.timer.start(self.update_interval)

    def stop_detection(self):
        """정지 버튼 클릭 시 탐지 중지"""
        if self.capture_active:
            self.capture_active = False
            self.timer.stop()
            self.log_message("탐지 정지")

    def on_conf_thresh_changed(self, value):
        self.conf_thresh = float(value)
        self.log_message(f"Confidence Threshold 변경: {self.conf_thresh}")

    def on_iou_thresh_changed(self, value):
        self.iou_thresh = float(value)
        self.log_message(f"IoU Threshold 변경: {self.iou_thresh}")

    def update_frame(self):
        """주기적으로 화면 캡쳐, 객체 탐지 및 UI 업데이트"""
        if not self.capture_active:
            return

        start_time = time.time()

        # 화면 캡쳐
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)  # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img0 = img.copy()

        # 전처리: letterbox (640x640) 후 텐서 변환
        img_resized = letterbox(img, new_shape=(640, 640))[0]
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_resized = np.ascontiguousarray(img_resized)
        img_tensor = torch.from_numpy(img_resized).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 모델 추론 & NMS (설정된 파라미터 사용)
        with torch.no_grad():
            pred = self.model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh)[0]

        # 객체 탐지 후 처리: 클래스별 탐지 정보 저장
        detections_by_class = {}
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred:
                cls_id = int(cls)
                label_name = self.model.names[cls_id]
                detection_info = {
                    'xyxy': [int(x) for x in xyxy],
                    'confidence': float(conf),
                    'label': label_name,
                    'x': int(xyxy[0])
                }
                detections_by_class.setdefault(label_name, []).append(detection_info)

        # 클래스별 최대 개수 제한 및 최종 탐지 리스트 구성
        final_detections = []
        for label, det_list in detections_by_class.items():
            det_list = sorted(det_list, key=lambda d: (-d['confidence'], d['x']))
            max_items = max_items_per_class.get(label, default_max_items)
            for idx, det in enumerate(det_list[:max_items], start=1):
                final_detections.append((label, idx, det))

        # 탐지 결과에 대해 박스 그리기 및 텍스트 오버레이
        for label, idx, det in final_detections:
            x1, y1, x2, y2 = det['xyxy']
            conf = det['confidence']
            text = f"{label} {conf:.2f}"
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img0, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 이미지 디스플레이 업데이트
        rgb_image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        # 탐지 객체 리스트 업데이트
        detection_texts = []
        for label, idx, det in final_detections:
            x1, y1, x2, y2 = det['xyxy']
            detection_texts.append(f"{label} [{idx}]: ({x1}, {y1}) {det['confidence']:.2f}")
        self.detection_list_edit.setPlainText("\n".join(detection_texts) if detection_texts else "No detections")

        # 그룹별 탐지 개수 계산 (접두사: main_, inGame_, inPlay_)
        # 그룹별 탐지 개수 계산 (접두사: main_, inGame_, inPlay_)
        group_counts = {"main": 0, "inGame": 0, "inPlay": 0}
        for label, _, _ in final_detections:
            if label.startswith("main_"):
                group_counts["main"] += 1
            elif label.startswith("inGame_"):
                group_counts["inGame"] += 1
            elif label.startswith("inPlay_"):
                group_counts["inPlay"] += 1

        self.counts_label.setText(
            f"Main: {group_counts['main']} | InGame: {group_counts['inGame']} | InPlay: {group_counts['inPlay']}"
        )

        # 상태 결정: 해당 그룹이 제일 많고, 동시에 최소 3개 이상일 때만 전환
        candidate_state = None
        candidate_count = 0
        if group_counts["main"] >= group_counts["inGame"] and group_counts["main"] >= group_counts["inPlay"]:
            candidate_state = "메인 화면"
            candidate_count = group_counts["main"]
        elif group_counts["inGame"] >= group_counts["inPlay"]:
            candidate_state = "인게임 화면"
            candidate_count = group_counts["inGame"]
        else:
            candidate_state = "인플레이 화면"
            candidate_count = group_counts["inPlay"]

        if candidate_count >= 3:
            new_state = candidate_state
        else:
            # 최소 조건 미달이면 상태 변경 없이 기존 상태 유지 (초기 상태는 "대기중"으로 가정)
            new_state = self.current_state

        if new_state != self.current_state:
            self.log_message(f"상태 전환: {self.current_state} → {new_state}")
            self.current_state = new_state

        self.state_label.setText(f"상태: {self.current_state}")

        # 성능 측정 (FPS 및 추론 시간)
        end_time = time.time()
        frame_time = (end_time - start_time) * 1000  # ms
        self.inference_time = frame_time
        dt = end_time - self.last_update_time
        self.last_update_time = end_time
        self.fps = 1 / dt if dt > 0 else 0
        self.fps_label.setText(f"FPS: {self.fps:.1f}")
        self.inf_time_label.setText(f"추론 시간: {self.inference_time:.1f} ms")

    def log_message(self, message):
        """로그 창에 메시지 추가 (탐지 시작 기준 시간 기준 상대 시간 포함)"""
        if hasattr(self, "start_time"):
            elapsed = time.time() - self.start_time
            # 경과 시간을 mm:ss 형식으로 표시
            mins, secs = divmod(int(elapsed), 60)
            timestamp = f"{mins:02d}:{secs:02d}"
        else:
            # 탐지가 시작되지 않았다면 현재 시스템 시간 사용
            timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_edit.append(f"[{timestamp}] {message}")


def main():
    parser = argparse.ArgumentParser(description="사용할 weights 설정")
    parser.add_argument('--weights', type=str, help='예: exp18')
    args = parser.parse_args()

    weights = f"runs/train/{args.weights}/weights/best.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights, map_location=device)
    model.eval()

    app = QApplication(sys.argv)
    window = MainWindow(model, device)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
