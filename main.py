import sys
import os
import importlib.util
import types
import warnings
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

# 경고 필터링
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
            
            # 현재 디렉토리 변경 (UI 파일 로드 문제 해결을 위해)
            original_dir = os.getcwd()
            os.chdir(self.qa_automation_dir)
            
            # 모듈 동적 임포트
            spec = importlib.util.spec_from_file_location("main", os.path.join(self.qa_automation_dir, "main.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 클래스 인스턴스 생성
            automation_window = module.mainWindow()
            self.tabs.addTab(automation_window, "QA-Automation Process")
            
            # 원래 디렉토리로 복원
            os.chdir(original_dir)
            
        except Exception as e:
            print(f"QA-Automation-Process 탭 로드 실패: {e}")
            error_widget = self.create_error_widget(f"QA-Automation-Process 로드 실패\n{str(e)}")
            self.tabs.addTab(error_widget, "QA-Automation Process (로드 실패)")

    def create_utils_package(self):
        """utils 패키지 및 하위 모듈 동적 생성"""
        # 모듈 생성 전에 이미 존재하는 모듈 제거
        for module_name in ['utils', 'utils.datasets', 'utils.general']:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # utils 패키지 생성
        utils_module = types.ModuleType('utils')
        sys.modules['utils'] = utils_module
        
        # utils/datasets.py 모듈 로드 및 등록
        utils_path = os.path.join(self.qa_video_dir, "utils")
        datasets_path = os.path.join(utils_path, "datasets.py")
        
        datasets_spec = importlib.util.spec_from_file_location('utils.datasets', datasets_path)
        datasets_module = importlib.util.module_from_spec(datasets_spec)
        sys.modules['utils.datasets'] = datasets_module
        datasets_spec.loader.exec_module(datasets_module)
        utils_module.datasets = datasets_module
        
        # utils/general.py 모듈 로드 및 등록
        general_path = os.path.join(utils_path, "general.py")
        general_spec = importlib.util.spec_from_file_location('utils.general', general_path)
        general_module = importlib.util.module_from_spec(general_spec)
        sys.modules['utils.general'] = general_module
        general_spec.loader.exec_module(general_module)
        utils_module.general = general_module
        
        # 다른 필요한 utils 하위 모듈도 동일하게 처리
        for submodule in ['torch_utils', 'plots', 'metrics']:
            module_path = os.path.join(utils_path, f"{submodule}.py")
            if os.path.exists(module_path):
                module_spec = importlib.util.spec_from_file_location(f'utils.{submodule}', module_path)
                module_obj = importlib.util.module_from_spec(module_spec)
                sys.modules[f'utils.{submodule}'] = module_obj
                module_spec.loader.exec_module(module_obj)
                setattr(utils_module, submodule, module_obj)

    def create_models_package(self):
        """models 패키지 및 experimental 모듈 등록"""
        # 모듈 생성 전에 이미 존재하는 모듈 제거
        for module_name in ['models', 'models.experimental']:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # models 패키지 생성
        models_module = types.ModuleType('models')
        sys.modules['models'] = models_module
        
        # models/experimental.py 모듈 로드 및 등록
        models_path = os.path.join(self.qa_video_dir, "models")
        experimental_path = os.path.join(models_path, "experimental.py")
        
        if os.path.exists(experimental_path):
            exp_spec = importlib.util.spec_from_file_location('models.experimental', experimental_path)
            exp_module = importlib.util.module_from_spec(exp_spec)
            sys.modules['models.experimental'] = exp_module
            exp_spec.loader.exec_module(exp_module)
            models_module.experimental = exp_module

    def load_video_process_tab(self):
        """QA-Video-Process 탭 로드"""
        try:
            # 경로 추가
            if self.qa_video_dir not in sys.path:
                sys.path.insert(0, self.qa_video_dir)
            
            # 모델 파일 확인
            model_path = os.path.join(self.qa_video_dir, "models", "yolov7.pt")
            if not os.path.exists(model_path):
                error_widget = self.create_error_widget("모델 파일이 없습니다. (models/yolov7.pt)")
                self.tabs.addTab(error_widget, "QA-Video-Process (모델 없음)")
                return
            
            # 현재 디렉토리 변경
            original_dir = os.getcwd()
            os.chdir(self.qa_video_dir)
            
            # utils 및 models 패키지 동적 생성
            self.create_utils_package()
            self.create_models_package()
            
            # detect_live.py 모듈 동적 임포트
            detect_live_path = os.path.join(self.qa_video_dir, "detect_live.py")
            with open(detect_live_path, 'r') as f:
                content = f.read()
            
            # 기존의 imports 유지 및 동적 로드
            module_spec = importlib.util.spec_from_file_location("detect_live", detect_live_path)
            detect_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(detect_module)
            
            # torch 및 모델 로드
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = sys.modules['models.experimental'].attempt_load(model_path, map_location=device)
            
            # MainWindow 클래스 생성
            video_window = detect_module.MainWindow(model, device)
            self.tabs.addTab(video_window, "QA-Video-Process")
            
            # 원래 디렉토리로 복원
            os.chdir(original_dir)
            
        except Exception as e:
            print(f"QA-Video-Process 탭 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            error_widget = self.create_error_widget(f"QA-Video-Process 로드 실패\n{str(e)}")
            self.tabs.addTab(error_widget, "QA-Video-Process (로드 실패)")

    def load_llm_tab(self):
        """AutoQA_llm 탭 로드"""
        try:
            # 경로 추가
            if self.autoqa_llm_dir not in sys.path:
                sys.path.insert(0, self.autoqa_llm_dir)
            
            # 현재 디렉토리 변경
            original_dir = os.getcwd()
            os.chdir(self.autoqa_llm_dir)
            
            # 모듈 동적 임포트
            spec = importlib.util.spec_from_file_location("main", os.path.join(self.autoqa_llm_dir, "main.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 클래스 인스턴스 생성
            llm_window = module.VideoPlayer()
            self.tabs.addTab(llm_window, "AutoQA_llm")
            
            # 원래 디렉토리로 복원
            os.chdir(original_dir)
            
        except Exception as e:
            print(f"AutoQA_llm 탭 로드 실패: {e}")
            error_widget = self.create_error_widget(f"AutoQA_llm 로드 실패\n{str(e)}")
            self.tabs.addTab(error_widget, "AutoQA_llm (로드 실패)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntegratedApp()
    window.show()
    sys.exit(app.exec_()) 