import sys
import os
import importlib.util
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

class IntegratedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoQA 통합 인터페이스")
        self.setGeometry(100, 100, 1600, 900)

        # 메인 위젯 및 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 탭 위젯 생성
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # 각 프로젝트 경로
        self.root_dir = os.getcwd()
        self.qa_automation_dir = os.path.join(self.root_dir, "QA-Automation-Process")
        self.qa_video_dir = os.path.join(self.root_dir, "QA-video-process")
        self.autoqa_llm_dir = os.path.join(self.root_dir, "AutoQA_llm")
        
        # 각 프로젝트의 창을 탭으로 추가 (순서대로)
        self.load_automation_process_tab()
        self.load_video_process_tab()
        self.load_llm_tab()

    def create_error_widget(self, message):
        """오류 메시지를 표시하는 위젯 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return widget

    def load_automation_process_tab(self):
        """QA-Automation-Process 탭 로드"""
        try:
            # 경로 추가
            if self.qa_automation_dir not in sys.path:
                sys.path.insert(0, self.qa_automation_dir)
            
            # 모듈 동적 임포트
            spec = importlib.util.spec_from_file_location("main", os.path.join(self.qa_automation_dir, "main.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 클래스 인스턴스 생성
            automation_window = module.mainWindow()
            self.tabs.addTab(automation_window, "QA-Automation Process")
            
        except Exception as e:
            print(f"QA-Automation-Process 탭 로드 실패: {e}")
            error_widget = self.create_error_widget(f"QA-Automation-Process 로드 실패\n{str(e)}")
            self.tabs.addTab(error_widget, "QA-Automation Process (로드 실패)")

    def load_video_process_tab(self):
        """QA-Video-Process 탭 로드"""
        try:
            # 경로 추가
            if self.qa_video_dir not in sys.path:
                sys.path.insert(0, self.qa_video_dir)
            
            # 필요한 모듈 동적 임포트
            spec = importlib.util.spec_from_file_location("detect_live", os.path.join(self.qa_video_dir, "detect_live.py"))
            detect_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(detect_module)
            
            # torch 및 모델 로드
            import torch
            
            # models 패키지 경로 추가
            models_path = os.path.join(self.qa_video_dir, "models")
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            
            # experimental 모듈 로드
            exp_spec = importlib.util.spec_from_file_location(
                "experimental", 
                os.path.join(models_path, "experimental.py")
            )
            experimental = importlib.util.module_from_spec(exp_spec)
            exp_spec.loader.exec_module(experimental)
            
            # 모델 로드
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = os.path.join(self.qa_video_dir, "models", "yolov7.pt")
            
            if os.path.exists(model_path):
                model = experimental.attempt_load(model_path, map_location=device)
                video_window = detect_module.MainWindow(model, device)
                self.tabs.addTab(video_window, "QA-Video-Process")
            else:
                error_widget = self.create_error_widget("모델 파일이 없습니다. (models/yolov7.pt)")
                self.tabs.addTab(error_widget, "QA-Video-Process (모델 없음)")
            
        except Exception as e:
            print(f"QA-Video-Process 탭 로드 실패: {e}")
            error_widget = self.create_error_widget(f"QA-Video-Process 로드 실패\n{str(e)}")
            self.tabs.addTab(error_widget, "QA-Video-Process (로드 실패)")

    def load_llm_tab(self):
        """AutoQA_llm 탭 로드"""
        try:
            # 경로 추가
            if self.autoqa_llm_dir not in sys.path:
                sys.path.insert(0, self.autoqa_llm_dir)
            
            # 모듈 동적 임포트
            spec = importlib.util.spec_from_file_location("main", os.path.join(self.autoqa_llm_dir, "main.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 클래스 인스턴스 생성
            llm_window = module.VideoPlayer()
            self.tabs.addTab(llm_window, "AutoQA_llm")
            
        except Exception as e:
            print(f"AutoQA_llm 탭 로드 실패: {e}")
            error_widget = self.create_error_widget(f"AutoQA_llm 로드 실패\n{str(e)}")
            self.tabs.addTab(error_widget, "AutoQA_llm (로드 실패)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntegratedApp()
    window.show()
    sys.exit(app.exec_()) 