import cv2
import numpy as np
import google.generativeai as genai
import os
import tempfile
import json
import logging
import sys
from PyQt5.QtCore import QThread, pyqtSignal
import shutil
import re
import time

try:
    from .sound_event_detector import SoundEffectLogService, load_expected_sound_events
except ImportError:
    from sound_event_detector import SoundEffectLogService, load_expected_sound_events

# 로그 설정
LOG_FORMAT = "[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 로거 인스턴스 생성
logger = logging.getLogger("Bug-Detector")
logger.setLevel(logging.DEBUG)

# 로그 핸들러 설정
file_handler = logging.FileHandler("bug_detector.log", mode="a")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
logger.addHandler(console_handler)

def log(message, level="DEBUG"):
    """
    로그 메시지 출력 - 로그 레벨에 따라 출력

    Args:
        message (str): 출력할 메시지
        level (str): 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)
    else:
        logger.debug(message)

# 모듈 초기화 로그
log("Bug Detector 모듈 초기화", "INFO")

class ScreenshotDetectionThread(QThread):
    """
    스크린샷 기반 버그 검출 스레드 클래스

    일정 시간마다 화면을 캡처하여 버그를 감지하는 스레드
    """
    screenshot_bug_detected = pyqtSignal(str, str, list) # 버그 감지 시그널 (버그 설명, 원본 이미지 경로, 바운딩 박스)
    status_update = pyqtSignal(str) # 상태 업데이트 시그널

    def __init__(self, api_key, interval=5):
        """
        초기화 함수

        Args:
            api_key (str): Google Gemini API 키
            interval (int, optional): 스크린샷 간격(초). 기본값 5.
        """
        super().__init__()
        self.api_key = api_key
        self.interval = interval
        self.running = False
        self.temp_dir = None

        # Gemini API 설정
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro-vision')
            log("Gemini API 초기화 성공", "INFO")
        except Exception as e:
            log(f"Gemini API 초기화 실패: {e}", "ERROR")
            self.model = None

    def run(self):
        """스레드 실행 함수 - 스크린샷 캡처 및 버그 감지"""
        log("스크린샷 감지 스레드 시작", "INFO")
        self.running = True

        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        log(f"임시 디렉토리 생성: {self.temp_dir}", "DEBUG")

        screenshot_count = 0

        while self.running:
            try:
                # 현재 시간 기록
                current_time = time.strftime("%Y%m%d_%H%M%S")
                screenshot_count += 1

                # 스크린샷 캡처
                self.status_update.emit(f"스크린샷 캡처 중 ({screenshot_count})...")
                screenshot = capture_screenshot()

                if screenshot is None:
                    log("스크린샷 캡처 실패", "ERROR")
                    self.status_update.emit("스크린샷 캡처 실패")
                    time.sleep(self.interval)
                    continue

                # 스크린샷 저장
                screenshot_path = os.path.join(self.temp_dir, f"screenshot_{current_time}.png")
                cv2.imwrite(screenshot_path, screenshot)
                log(f"스크린샷 저장: {screenshot_path}", "DEBUG")

                # 버그 감지
                self.status_update.emit("버그 분석 중...")
                bug_result = self.detect_bugs_in_image(screenshot_path)

                if bug_result:
                    bug_description, confidence = bug_result
                    log(f"버그 감지됨: {bug_description} (신뢰도: {confidence})", "INFO")

                    # 결과 이미지 생성 및 시그널 발생
                    bounding_boxes = []  # 이 예제에서는 바운딩 박스 정보가 없음
                    self.screenshot_bug_detected.emit(bug_description, screenshot_path, bounding_boxes)

                    # 일정 시간 대기 (버그 감지 후 더 긴 대기 시간)
                    self.status_update.emit(f"버그 감지됨: {bug_description[:50]}...")
                    time.sleep(self.interval * 2)
                else:
                    self.status_update.emit("버그 없음, 다음 스크린샷 대기 중...")
                    time.sleep(self.interval)

            except Exception as e:
                log(f"스크린샷 분석 중 오류 발생: {e}", "ERROR")
                self.status_update.emit(f"오류 발생: {str(e)}")
                time.sleep(self.interval)

        # 스레드 종료 시 임시 디렉토리 정리
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                log(f"임시 디렉토리 삭제: {self.temp_dir}", "DEBUG")
            except Exception as e:
                log(f"임시 디렉토리 삭제 실패: {e}", "ERROR")

def capture_screenshot():
    """
    전체 화면 스크린샷 캡처

    Returns:
        numpy.ndarray: 캡처된 이미지 또는 실패 시 None
    """
    try:
        # OpenCV로 스크린샷 캡처
        import pyautogui

        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)  # PIL -> OpenCV 색상 변환
        return screenshot
    except Exception as e:
        log(f"스크린샷 캡처 실패: {e}", "ERROR")
        return None

class BugDetector(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    log_updated = pyqtSignal(str)  # 로그 업데이트 시그널
    bug_detected = pyqtSignal(dict)  # 버그 감지 시그널

    def __init__(self, video_path, video_info=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.video_info = video_info or {}  # 동영상 정보 저장
        self.logs = []  # API 응답 로그 저장
        self.sound_effect_service = SoundEffectLogService()
        self.sound_effect_logs = []
        log(f"BugDetector 초기화 - 비디오 경로: {video_path}", "INFO")

        self.persona_prompt = """당신은 숙련된 게임 QA 테스터입니다. 주어진 게임 플레이 영상을 분석하여 버그를 찾아내는것이 목표입니다. 단순히 한 장면만 보지 말고, 영상 전체의 흐름과 맥락 속에서 게임의 규칙이나 일반적인 상식을 벗어나는 현상을 식별해야 합니다.

[보고 형식]
버그를 발견했다면, 분석 결과에 대해 다음 필드를 포함하는 Json 형식으로 응답하세요.

- time: 각 버그가 명확하게 나타나는 시작 시간. (분:초.밀리초: 0:20.35)
- frame: 버그가 시작하는 프레임을 함께 기재해야 합니다. (프레임 번호: 50)
- type: 각 상위 항목에 대한 하위 항목 선택. (상위항목-하위항목)
    - Physics – Clipping, Stuck, Ragdoll, Levitation, Sinking, Jitter
    - Texture – Missing texture, Missing material, Visible corruption, Visible Stretching, Visible Seam
    - Modeling – Vertex spike, Vertex Tear, Missing
    - Animation – T-pose, A-pose, Frozen, Joint hyperextension, Sliding
    - Sound – Missing sound, Loop, Noise, Abrupt Volume Change, Desync
    - UI – Overlap, Incorrect data display, Text overflow, Visual layout
    - AI – Pathfinding failure, Non-reactive
    - Rendering – LOD, Shadow, Lighting, Tearing, Object culling, Z-fighting, Draw Order
    - Camera – Camera clipping, Erratic movement
    - Performance – Fps drop, Stuttering, Freezing, Visible loading delay

-description: 어디서 (화면의 어느 부분), 무엇이 (어떤 객체나 요소가), 어떻게 이상한지를 객관적이고 간단하게 기술합니다.

[주의사항]
분석은 객관적이어야 하며, 게임의 장르, 디자인, 의도, 재미 등에 대한 주관적인 평가는 포함하지 않습니다.
버그가 존재하는지 주어진 모든 타입별로 고려합니다.
오직 시각적으로 확인 가능한 명백한 이상 현상만을 버그로 보고합니다. 화면에 보이는 객체의 종류(2D, 3D 등)나 게임 규칙을 미리 가정하지 마십시오. 게임 시스템 내부 로직의 오류는 추측하지 않습니다.
버그는 여러개가 있을 수 있습니다.
제공된 영상과 추가 정보만을 사용합니다.

[시작]
이제 분석할 게임 이미지 또는 비디오를 제공해주세요. 보고서 형식에 맞춰 결과를 알려드리겠습니다.
   """
        log("BugDetector 프롬프트 초기화 완료", "DEBUG")

    def run(self):
        """비디오 프레임 분석 및 버그 감지"""
        log("BugDetector 실행 시작", "INFO")
        temp_path = None
        try:
            # 비디오 파일 유효성 검사
            if not os.path.exists(self.video_path):
                error_msg = "비디오 파일을 찾을 수 없습니다."
                log(error_msg, "ERROR")
                raise Exception(error_msg)

            # 비디오 파일 크기 확인
            file_size = os.path.getsize(self.video_path)
            if file_size > 100 * 1024 * 1024:  # 100MB 제한
                error_msg = "비디오 파일이 너무 큽니다. 100MB 이하의 파일을 사용해주세요."
                log(error_msg, "ERROR")
                raise Exception(error_msg)

            # 비디오 파일 검증
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                error_msg = "비디오 파일을 열 수 없습니다."
                log(error_msg, "ERROR")
                raise Exception(error_msg)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if total_frames <= 0 or fps <= 0:
                error_msg = "잘못된 비디오 형식입니다."
                log(error_msg, "ERROR")
                raise Exception(error_msg)

            log(f"비디오 파일 검증 완료 - 총 프레임: {total_frames}, FPS: {fps}", "INFO")
            cap.release()

            # Gemini API 키 설정 및 모델 초기화
            model = None
            try:
                api_key = (
                    self.video_info.get("api_key")
                    or os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                )
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
                    log("Gemini API 초기화 성공", "INFO")
                else:
                    log("Gemini API 키가 없어 API 분석은 건너뜁니다.", "WARNING")
            except Exception as e:
                log(f"Gemini API 초기화 실패, 로컬 분석만 진행: {str(e)}", "WARNING")
                model = None

            # 비디오 파일을 임시 파일로 복사
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
                try:
                    shutil.copy2(self.video_path, temp_path)
                    log(f"비디오 파일을 임시 파일로 복사 - 경로: {temp_path}", "DEBUG")
                except Exception as e:
                    error_msg = f"비디오 파일 복사 중 오류 발생: {str(e)}"
                    log(error_msg, "ERROR")
                    raise Exception(error_msg)

            # 비디오 파일을 바이너리로 읽기
            try:
                with open(temp_path, 'rb') as f:
                    video_data = f.read()
                log(f"비디오 데이터 읽기 완료 - 크기: {len(video_data)/(1024*1024):.2f}MB", "DEBUG")
            except Exception as e:
                error_msg = f"비디오 파일 읽기 중 오류 발생: {str(e)}"
                log(error_msg, "ERROR")
                raise Exception(error_msg)

            # 실제 오디오 데이터에서 사운드 이펙트 로그를 먼저 추출한다.
            sound_effect_logs = []
            sound_effect_summary = {}
            sound_validation = []
            try:
                expected_events = []
                expected_path = self.video_info.get("expected_sound_events_path")
                if expected_path:
                    expected_events = load_expected_sound_events(expected_path)
                elif isinstance(self.video_info.get("expected_sound_events"), list):
                    expected_events = self.video_info["expected_sound_events"]

                sound_package = self.sound_effect_service.extract_with_validation(
                    self.video_path,
                    fps=fps,
                    expected_events=expected_events,
                )
                sound_effect_logs = sound_package["events"]
                sound_effect_summary = sound_package["summary"]
                sound_validation = sound_package["validation"]
                self.sound_effect_logs = sound_effect_logs
                self.log_updated.emit(
                    f"로컬 사운드 이펙트 로그 {len(sound_effect_logs)}개 추출 "
                    f"(검증 이슈 {len([item for item in sound_validation if item.get('status') != 'matched'])}개)"
                )
                log(f"로컬 사운드 이펙트 로그 추출 완료: {len(sound_effect_logs)}개", "INFO")
            except Exception as e:
                self.log_updated.emit(f"로컬 사운드 이펙트 로그 추출 건너뜀: {str(e)}")
                log(f"로컬 사운드 이펙트 로그 추출 실패: {str(e)}", "WARNING")

            # 초기 프롬프트 준비
            additional_info = []
            additional_info.append(f"비디오 정보:\n- 총 프레임: {total_frames}\n- FPS: {fps}")
            if sound_effect_logs:
                additional_info.append("로컬 사운드 이펙트 요약:\n" + json.dumps(sound_effect_summary, ensure_ascii=False, indent=2))
                additional_info.append("로컬 사운드 이펙트 로그:\n" + json.dumps(sound_effect_logs, ensure_ascii=False, indent=2))
            if sound_validation:
                additional_info.append("사운드 기대값 비교 결과:\n" + json.dumps(sound_validation, ensure_ascii=False, indent=2))

            # 사용자가 입력한 정보 추가
            if self.video_info:
                if 'game_info' in self.video_info and self.video_info['game_info']:
                    additional_info.append(f"게임 정보:\n{self.video_info['game_info']}")
                if 'video_description' in self.video_info and self.video_info['video_description']:
                    additional_info.append(f"동영상 설명:\n{self.video_info['video_description']}")

            # 프롬프트에 추가 정보 결합
            additional_info_text = "\n\n".join(additional_info)
            initial_prompt = f"{self.persona_prompt}\n\n{additional_info_text}"

            self.log_updated.emit("초기 프롬프트 준비 완료")
            log("초기 프롬프트 준비 완료", "INFO")
            self.progress.emit(10)

            # 프롬프트와 비디오 데이터 준비
            prompt_parts = [
                initial_prompt,
                {'mime_type': 'video/mp4', 'data': video_data}
            ]

            # Gemini API로 분석
            if model is None:
                self.log_updated.emit("Gemini API 키가 없어 비디오 API 분석은 건너뛰고 로컬 로그만 사용합니다.")
                log("Gemini API 분석 건너뜀", "WARNING")
            else:
                self.log_updated.emit("비디오 분석 시작...")
                log("Gemini API 분석 시작", "INFO")
            self.progress.emit(20)

            try:
                response = None
                if model is not None:
                    response = model.generate_content(
                        prompt_parts,
                        request_options={"timeout": 600}  # 10분 타임아웃
                    )

                    if not response or not hasattr(response, 'text'):
                        raise Exception("API 응답이 올바르지 않습니다.")

                    self.log_updated.emit("API 응답 수신 완료")
                    self.progress.emit(40)

                    # API 응답 로그 저장
                    self.log_updated.emit("API 응답:\n" + response.text)
                    print("API 응답:", response.text)  # 콘솔에도 출력

                # 응답 파싱
                if response and hasattr(response, 'text'):
                    response_text = response.text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]  # ```json 제거
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]  # ``` 제거
                    response_text = response_text.strip()

                    try:
                        # JSON 응답 파싱
                        bug_data = json.loads(response_text)
                        if not isinstance(bug_data, list):
                            bug_data = [bug_data]

                        total_bugs = len(bug_data)
                        self.log_updated.emit(f"감지된 버그 수: {total_bugs}")

                        for i, bug in enumerate(bug_data, 1):
                            try:
                                # 진행률 업데이트 (50%~90%)
                                progress = 50 + int((i / total_bugs) * 40)
                                self.progress.emit(progress)

                                # 버그 데이터 추출
                                time = bug.get('time', '전체 비디오')
                                frame = int(bug.get('frame', '0'))
                                bug_type = ', '.join(bug.get('type', ['알 수 없는 유형']))
                                description = bug.get('description', '설명 없음')

                                # 해당 프레임의 스크린샷 캡처
                                image = None
                                if frame >= 0:
                                    try:
                                        cap = cv2.VideoCapture(self.video_path)
                                        if cap.isOpened():
                                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                                            ret, frame_img = cap.read()
                                            if ret:
                                                # 이미지 크기 조정 (메모리 사용량 감소)
                                                height, width = frame_img.shape[:2]
                                                max_size = 800
                                                if width > max_size or height > max_size:
                                                    scale = max_size / max(width, height)
                                                    frame_img = cv2.resize(frame_img, None, fx=scale, fy=scale)

                                                # 이미지를 바이트로 변환 (압축 품질 조정)
                                                _, buffer = cv2.imencode('.jpg', frame_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                                image = buffer.tobytes()
                                    except Exception as e:
                                        self.log_updated.emit(f"스크린샷 캡처 중 오류 발생: {str(e)}")
                                    finally:
                                        if 'cap' in locals():
                                            cap.release()

                                # 버그 데이터 생성 및 전송
                                bug_data = {
                                    'time': time,
                                    'frame': frame,
                                    'type': bug_type,
                                    'description': description,
                                    'image': image,
                                    'full_response': json.dumps(bug, indent=2, ensure_ascii=False)
                                }
                                self.bug_detected.emit(bug_data)
                                self.log_updated.emit(f"버그 감지됨: 시간={time}, 프레임={frame}, 유형={bug_type}")

                            except Exception as e:
                                self.log_updated.emit(f"버그 데이터 파싱 중 오류 발생: {str(e)}")
                                continue
                    except json.JSONDecodeError as e:
                        self.log_updated.emit(f"JSON 파싱 오류: {str(e)}")
                        raise Exception("API 응답이 올바른 JSON 형식이 아닙니다.")

                for sound_event in sound_effect_logs:
                    sound_bug_data = {
                        'time': sound_event.get('time', '전체 비디오'),
                        'frame': int(sound_event.get('frame', 0)),
                        'type': f"Sound - {sound_event.get('type', 'Sound Effect')}",
                        'description': (
                            f"{sound_event.get('description', '')} "
                            f"direction={sound_event.get('direction', 'unknown')}"
                        ).strip(),
                        'image': None,
                        'full_response': json.dumps(sound_event, indent=2, ensure_ascii=False)
                    }
                    self.bug_detected.emit(sound_bug_data)
                    self.log_updated.emit(
                        f"사운드 로그 추가: 시간={sound_bug_data['time']}, "
                        f"프레임={sound_bug_data['frame']}, 유형={sound_bug_data['type']}"
                    )

                for issue in sound_validation:
                    if issue.get("status") == "matched":
                        continue
                    expected = issue.get("expected") or {}
                    observed = issue.get("observed") or {}
                    issue_frame = int(observed.get("frame", 0))
                    if not observed and expected.get("start_seconds") is not None:
                        issue_frame = int(round(float(expected.get("start_seconds", 0.0)) * fps))
                    validation_bug_data = {
                        'time': observed.get('time') or expected.get('time', '전체 비디오'),
                        'frame': issue_frame,
                        'type': f"Sound - {issue.get('status', 'validation_issue')}",
                        'description': issue.get('description', ''),
                        'image': None,
                        'full_response': json.dumps(issue, indent=2, ensure_ascii=False)
                    }
                    self.bug_detected.emit(validation_bug_data)
                    self.log_updated.emit(
                        f"사운드 검증 이슈: 시간={validation_bug_data['time']}, "
                        f"유형={validation_bug_data['type']}"
                    )

                # 임시 파일 삭제
                os.remove(temp_path)

                # 완료 시그널 발생
                self.progress.emit(100)
                self.finished.emit([])
                self.log_updated.emit("분석이 완료되었습니다.")

            except Exception as e:
                self.log_updated.emit(f"API 분석 중 오류 발생: {str(e)}")
                print(f"Gemini API 분석 중 오류 발생: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                self.progress.emit(100)  # 오류 발생 시에도 진행률 100%로 설정
                self.finished.emit([])

        except Exception as e:
            self.log_updated.emit(f"비디오 처리 중 오류 발생: {str(e)}")
            print(f"비디오 처리 중 오류 발생: {str(e)}")
            self.progress.emit(100)
            self.finished.emit([])
