from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame, 
                            QScrollArea, QHBoxLayout, QTextEdit, QPushButton, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QImage
import cv2
import numpy as np
import json

class BugItem(QWidget):
    clicked = pyqtSignal(int)  # 프레임 번호를 전달하는 시그널
    
    def __init__(self, frame, time, bug_type, description, image=None, full_response='', parent=None):
        super().__init__(parent)
        self.frame = frame
        self.time = time
        self.bug_type = bug_type
        self.description = description
        self.image = image
        self.full_response = full_response
        
        self.init_ui()
        
    def init_ui(self):
        try:
            layout = QHBoxLayout()
            
            # 이미지 표시
            if self.image is not None:
                try:
                    # 바이트 문자열을 numpy 배열로 변환
                    nparr = np.frombuffer(self.image, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # 이미지 크기 조정 (최대 200x150)
                        h, w = frame.shape[:2]
                        scale = min(200/w, 150/h)
                        new_w, new_h = int(w*scale), int(h*scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = frame.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                        
                        self.image_label = QLabel()
                        self.image_label.setPixmap(pixmap)
                        self.image_label.setFixedSize(new_w, new_h)
                        self.image_label.setStyleSheet("border: 1px solid #ccc;")
                        layout.addWidget(self.image_label)
                except Exception as e:
                    print(f"이미지 처리 중 오류 발생: {str(e)}")
            
            # 버그 정보 표시
            info_layout = QVBoxLayout()
            
            # 시간 정보
            time_label = QLabel(f"시간: {self.time}")
            time_label.setStyleSheet("font-weight: bold; color: #333;")
            info_layout.addWidget(time_label)
            
            # 프레임 정보
            frame_label = QLabel(f"프레임: {self.frame}")
            frame_label.setStyleSheet("color: #666;")
            info_layout.addWidget(frame_label)
            
            # 버그 유형
            type_label = QLabel(f"유형: {self.bug_type}")
            type_label.setStyleSheet("color: #666;")
            info_layout.addWidget(type_label)
            
            # 버그 설명
            desc_label = QLabel("설명:")
            desc_label.setStyleSheet("font-weight: bold; color: #333;")
            info_layout.addWidget(desc_label)
            
            desc_text = QTextEdit()
            desc_text.setPlainText(self.description)
            desc_text.setReadOnly(True)
            desc_text.setMaximumHeight(100)
            desc_text.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
            info_layout.addWidget(desc_text)
            
            # 전체 응답 보기 버튼
            if self.full_response:
                show_full_btn = QPushButton("상세 정보 보기")
                show_full_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2196F3;
                        color: white;
                        border: none;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #1976D2;
                    }
                """)
                show_full_btn.clicked.connect(self.show_full_response)
                info_layout.addWidget(show_full_btn)
            
            layout.addLayout(info_layout)
            self.setLayout(layout)
            
            # 클릭 이벤트 연결
            self.mousePressEvent = self.on_click
            
            # 위젯 스타일 설정
            self.setStyleSheet("""
                QWidget {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 5px;
                }
                QWidget:hover {
                    background-color: #f9f9f9;
                }
            """)
            
        except Exception as e:
            print(f"버그 아이템 UI 초기화 중 오류 발생: {str(e)}")
            # 최소한의 UI는 표시
            layout = QVBoxLayout()
            error_label = QLabel(f"오류 발생: {str(e)}")
            layout.addWidget(error_label)
            self.setLayout(layout)
        
    def on_click(self, event):
        self.clicked.emit(self.frame)
        
    def show_full_response(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("상세 정보")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout()
        
        # JSON 데이터를 보기 좋게 표시
        try:
            json_data = json.loads(self.full_response)
            formatted_text = json.dumps(json_data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            formatted_text = self.full_response
        
        text_edit = QTextEdit()
        text_edit.setPlainText(formatted_text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        layout.addWidget(text_edit)
        
        dialog.setLayout(layout)
        dialog.exec_()

class BugListWidget(QWidget):
    bug_clicked = pyqtSignal(int)  # 프레임 번호를 전달하는 시그널
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 스크롤 영역 생성
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # 버그 목록을 담을 위젯
        self.bug_list_widget = QWidget()
        self.bug_list_layout = QVBoxLayout(self.bug_list_widget)
        self.bug_list_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.bug_list_widget)
        layout.addWidget(scroll)
        
        self.setLayout(layout)
        
    def add_bug(self, frame, time, bug_type, description, image=None, full_response=''):
        """새로운 버그 아이템 추가"""
        try:
            print(f"BugListWidget.add_bug 호출 - 프레임: {frame}, 시간: {time}")  # 디버깅 로그
            bug_item = BugItem(frame, time, bug_type, description, image, full_response)
            bug_item.clicked.connect(self.bug_clicked.emit)  # 버그 아이템 클릭 시그널 연결
            self.bug_list_layout.addWidget(bug_item)
            print("BugItem이 레이아웃에 추가됨")  # 디버깅 로그
            return bug_item
        except Exception as e:
            print(f"버그 아이템 추가 중 오류 발생: {str(e)}")
            return None
    
    def clear_bugs(self):
        """버그 목록 초기화"""
        # 레이아웃의 모든 위젯 제거
        while self.bug_list_layout.count():
            item = self.bug_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # 버그 아이템 목록 초기화
        self.bug_items = [] 