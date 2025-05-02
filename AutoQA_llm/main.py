import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QSlider, 
                            QLabel, QStyle, QMenuBar, QMenu, QAction, QScrollArea, 
                            QInputDialog, QProgressDialog, QMessageBox, QTextEdit, 
                            QDialog, QLineEdit, QDialogButtonBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from bug_list import BugListWidget
from bug_detector import BugDetector

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("동영상 플레이어")
        self.setGeometry(100, 100, 1200, 800)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        
        # 왼쪽 패널 (비디오 플레이어)
        left_panel = QVBoxLayout()
        
        # 비디오 레이블
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        left_panel.addWidget(self.video_label)
        
        # 컨트롤 버튼
        control_layout = QHBoxLayout()
        
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_pause)
        control_layout.addWidget(self.play_button)
        
        self.prev_frame_button = QPushButton("이전 프레임")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        control_layout.addWidget(self.prev_frame_button)
        
        self.next_frame_button = QPushButton("다음 프레임")
        self.next_frame_button.clicked.connect(self.next_frame)
        control_layout.addWidget(self.next_frame_button)
        
        self.prev_5sec_button = QPushButton("5초 전")
        self.prev_5sec_button.clicked.connect(self.prev_5sec)
        control_layout.addWidget(self.prev_5sec_button)
        
        self.next_5sec_button = QPushButton("5초 후")
        self.next_5sec_button.clicked.connect(self.next_5sec)
        control_layout.addWidget(self.next_5sec_button)
        
        left_panel.addLayout(control_layout)
        
        # 위치 슬라이더
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        left_panel.addWidget(self.position_slider)
        
        # 프레임 정보 레이블
        self.frame_info_label = QLabel("프레임: 0 / 0")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.frame_info_label)
        
        # 오른쪽 패널 (버그 목록)
        right_panel = QVBoxLayout()
        
        # 버그 목록
        self.bug_list = BugListWidget()
        self.bug_list.bug_clicked.connect(self.jump_to_frame)  # 버그 클릭 시그널 연결
        right_panel.addWidget(self.bug_list)
        
        # 로그 텍스트 에디트
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        right_panel.addWidget(self.log_text)
        
        # 메인 레이아웃에 패널 추가
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        # 중앙 위젯 설정
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 상태 변수 초기화
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        # 메뉴바 설정
        self.setup_menubar()
        
        # 스타일 설정
        self.setup_styles()
        
    def setup_menubar(self):
        # 메뉴바 설정
        menubar = self.menuBar()
        file_menu = menubar.addMenu('파일')
        
        open_action = QAction('열기', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        exit_action = QAction('종료', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def setup_styles(self):
        # 스타일 설정
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #e0e0e0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #1976D2;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1976D2;
            }
            QLabel {
                color: #333;
                font-size: 14px;
            }
        """)
        
    def open_file(self):
        """비디오 파일 열기"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "비디오 파일 선택", "", 
                "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
            )
            
            if file_name:
                # 동영상 정보 입력 다이얼로그 표시
                video_info = self.get_video_info()
                if not video_info:  # 사용자가 취소한 경우
                    return
                
                # 기존 비디오 캡처 객체 해제
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()
                
                # 새로운 비디오 캡처 객체 생성
                self.cap = cv2.VideoCapture(file_name)
                if not self.cap.isOpened():
                    raise Exception("비디오 파일을 열 수 없습니다.")
                
                # 비디오 정보 가져오기
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.video_path = file_name  # 비디오 경로 저장
                
                # 입력받은 정보 저장
                self.game_info = video_info.get('game_info', '')
                self.video_description = video_info.get('video_description', '')
                
                # UI 업데이트
                self.position_slider.setRange(0, self.total_frames - 1)
                self.position_slider.setValue(0)
                self.frame_info_label.setText(f"프레임: 0 / {self.total_frames - 1}")
                
                # 0번째 프레임으로 이동하고 표시
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled_pixmap = pixmap.scaled(
                        self.video_label.size(), 
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.video_label.setPixmap(scaled_pixmap)
                
                # 버그 감지 시작
                self.start_bug_detection()
                
                # 창 제목 업데이트
                self.setWindowTitle(f"동영상 플레이어 - {file_name.split('/')[-1]}")
                
        except Exception as e:
            print(f"비디오 파일 열기 중 오류 발생: {str(e)}")
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            self.cap = None
            self.frame_info_label.setText("비디오 로드 실패")
            self.position_slider.setValue(0)
            self.position_slider.setMaximum(0)
            
    def get_video_info(self):
        """동영상 정보 입력 다이얼로그 표시"""
        dialog = QDialog(self)
        dialog.setWindowTitle("동영상 정보 입력")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        # 게임 정보 섹션
        game_group = QGroupBox("게임 정보")
        game_layout = QVBoxLayout()
        
        game_info_edit = QTextEdit()
        game_info_edit.setMaximumHeight(100)
        game_info_edit.setPlaceholderText("게임에 대한 정보를 입력하세요")
        game_layout.addWidget(game_info_edit)
        
        game_group.setLayout(game_layout)
        layout.addWidget(game_group)
        
        # 동영상 설명 섹션
        desc_group = QGroupBox("동영상 설명")
        desc_layout = QVBoxLayout()
        
        video_desc_edit = QTextEdit()
        video_desc_edit.setMaximumHeight(100)
        video_desc_edit.setPlaceholderText("동영상의 주요 내용이나 특징을 설명하세요")
        desc_layout.addWidget(video_desc_edit)
        
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)
        
        # 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            return {
                'game_info': game_info_edit.toPlainText().strip(),
                'video_description': video_desc_edit.toPlainText().strip()
            }
        return None
    
    def start_bug_detection(self):
        """버그 감지 시작"""
        try:
            # 버그 목록 초기화
            self.bug_list.clear_bugs()
            
            # 사용자가 입력한 동영상 정보 가져오기
            video_info = {}
            if hasattr(self, 'game_info'):
                video_info['game_info'] = self.game_info
            if hasattr(self, 'video_description'):
                video_info['video_description'] = self.video_description
            
            # 버그 감지 스레드 시작
            self.bug_detector = BugDetector(self.video_path, video_info)
            self.bug_detector.progress.connect(self.update_progress)
            self.bug_detector.finished.connect(self.bug_detection_finished)
            self.bug_detector.log_updated.connect(self.update_log)
            self.bug_detector.bug_detected.connect(self.add_detected_bugs)
            
            # 진행 상태 다이얼로그 표시
            self.progress_dialog = QProgressDialog("버그 감지 중...", "취소", 0, 100, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.canceled.connect(self.cancel_bug_detection)
            self.progress_dialog.show()
            
            # 버그 감지 시작
            self.bug_detector.start()
            
        except Exception as e:
            print(f"버그 감지 시작 중 오류 발생: {str(e)}")
            
    def update_progress(self, value):
        """진행 상태 업데이트"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(value)
            
    def bug_detection_finished(self, bugs):
        """버그 감지 완료"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            
    def cancel_bug_detection(self):
        """버그 감지 취소"""
        if hasattr(self, 'bug_detector'):
            self.bug_detector.quit()
            self.bug_detector.wait()
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
    
    def update_log(self, log_text):
        """로그 업데이트"""
        try:
            self.log_text.append(log_text)
            # 스크롤을 맨 아래로 이동
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        except Exception as e:
            print(f"로그 업데이트 중 오류 발생: {str(e)}")

    def add_detected_bugs(self, bug_data):
        """감지된 버그를 목록에 추가"""
        try:
            print(f"버그 데이터 수신: {bug_data}")  # 디버깅 로그
            if bug_data:
                # 현재 프레임의 이미지 캡처
                if self.cap is not None:
                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, bug_data.get('frame', current_frame))
                    ret, frame = self.cap.read()
                    if ret:
                        # 이미지를 JPEG 형식으로 인코딩
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_data = buffer.tobytes()
                    else:
                        image_data = None
                    # 원래 프레임 위치로 돌아가기
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                else:
                    image_data = None
                
                # 버그 데이터에서 필요한 정보 추출
                frame = bug_data.get('frame', 0)
                time = bug_data.get('time', '전체 비디오')
                bug_type = str(bug_data.get('type', '알 수 없는 버그'))  # 문자열로 변환하여 처리
                description = bug_data.get('description', '설명 없음')
                full_response = bug_data.get('full_response', '')
                
                print(f"추출된 버그 정보 - 시간: {time}, 유형: {bug_type}, 프레임: {frame}")  # 디버깅 로그
                
                # 버그 위젯에 추가
                try:
                    bug_item = self.bug_list.add_bug(
                        frame=frame,
                        time=time,
                        bug_type=bug_type,
                        description=description,
                        image=image_data,
                        full_response=full_response
                    )
                    
                    if bug_item:
                        print(f"버그 위젯이 성공적으로 추가됨 - 시간: {time}, 유형: {bug_type}")  # 디버깅 로그
                    else:
                        print("버그 위젯 추가 실패")  # 디버깅 로그
                except Exception as e:
                    print(f"버그 위젯 생성 중 오류 발생: {str(e)}")
                    
        except Exception as e:
            print(f"버그 추가 중 오류 발생: {str(e)}")
    
    def update_frame(self):
        if self.cap is not None and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                self.update_frame_info()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.is_playing = False
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
    
    def update_frame_info(self):
        if self.cap is not None:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.position_slider.setValue(current_frame)
            self.frame_info_label.setText(f"프레임: {current_frame} / {self.total_frames}")
    
    def set_position(self, position):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                self.update_frame_info()
    
    def play_pause(self):
        if self.cap is not None:
            if self.is_playing:
                self.timer.stop()
                self.is_playing = False
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            else:
                self.timer.start(int(1000 / self.fps))
                self.is_playing = True
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
    
    def prev_frame(self):
        if self.cap is not None:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(0, current_frame - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                self.update_frame_info()
    
    def next_frame(self):
        if self.cap is not None:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = min(self.total_frames - 1, current_frame + 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                self.update_frame_info()
    
    def prev_5sec(self):
        if self.cap is not None:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(0, current_frame - int(5 * self.fps))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                self.update_frame_info()
    
    def next_5sec(self):
        if self.cap is not None:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = min(self.total_frames - 1, current_frame + int(5 * self.fps))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                self.update_frame_info()
    
    def jump_to_frame(self, frame_number):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                self.update_frame_info()
    
    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_()) 