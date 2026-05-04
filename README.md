# AutonomousQA Tool

AutonomousQA Tool은 게임 QA 자동화를 위한 실험/운영 통합 저장소입니다. 현재 저장소에는 PyQt5 기반 매크로 자동화, GUI 템플릿 매칭, OCR, YOLO 실시간 탐지, 비디오 기반 LLM 버그 분석, 사운드 이펙트 로그 검증 기능이 함께 들어 있습니다.

주요 테스트 대상은 Geometry Dash 계열 게임 플레이이며, Windows 데스크톱 환경을 1차 실행 환경으로 가정합니다. WSL/Linux에서도 일부 PyQt5 UI와 비디오 분석 기능은 실행할 수 있지만, 실제 창 제어와 입력 자동화는 Windows 환경에서 확인해야 합니다.

## 핵심 기능

| 영역 | 설명 | 대표 파일 |
| --- | --- | --- |
| GUI 매크로 자동화 | 실행 중인 게임 창을 선택하고, UI 템플릿 매칭 결과에 따라 반복 액션을 수행합니다. | `main.py`, `QA-Auto/` |
| OCR/템플릿 인식 | OpenCV 템플릿 매칭과 OCR로 게임 UI 상태를 식별합니다. | `utils/`, `QA-Auto/utils/` |
| YOLO 실시간 탐지 | 화면 캡처를 YOLOv7 모델로 분석해 게임 상태와 객체를 표시합니다. | `detect_live.py`, `models/`, `utils_yolo/` |
| LLM 비디오 QA | 게임 플레이 비디오를 로드하고 Gemini 기반 버그 분석 결과를 버그 목록에 표시합니다. | `AutoQA_llm/main.py`, `AutoQA_llm/bug_detector.py` |
| 사운드 이펙트 로그 | 실제 비디오 오디오에서 사운드 이벤트를 추출하고 기대 이벤트 JSON과 비교합니다. | `AutoQA_llm/sound_event_detector.py` |

## 디렉터리 구조

```text
.
├── main.py                         # 통합 AutoQA GUI 진입점
├── detect_live.py                  # YOLOv7 실시간 화면 탐지 GUI
├── requirements.txt                # 루트 통합 실행 의존성
├── AutoQA_llm/                     # 비디오/LLM/사운드 기반 QA 도구
│   ├── main.py                     # 비디오 플레이어 + 버그 분석 UI
│   ├── bug_detector.py             # Gemini 분석 및 로컬 사운드 로그 결합
│   ├── sound_event_detector.py     # FFmpeg 기반 로컬 오디오 이벤트 추출
│   └── test_sound_event_detector.py
├── QA-Auto/                        # 이전 AutoQA 데스크톱 앱 패키지
├── QA_video_process/               # 비디오 처리 실험 코드
├── docs/
│   └── sound_expected_events.example.json
├── models/, utils_yolo/            # YOLOv7 모델/유틸리티
└── images/, screen/                # README 이미지와 런타임 화면 자산
```

## 설치

이 저장소는 이미지, 모델, 영상 등 큰 파일이 많아 Git LFS 사용을 전제로 합니다. 새로 클론한 뒤 먼저 LFS 파일을 받습니다.

```sh
git lfs install
git lfs pull
```

Python은 3.10 계열을 권장합니다. 현재 루트는 `uv` 기반 단일 가상환경으로 설치할 수 있도록 구성되어 있습니다.

```sh
python -m pip install --user uv
uv sync
uv run autoqa-check
```

기본 `uv sync`는 PyQt5 GUI, LLM 비디오 QA, 사운드 로그 추출에 필요한 공통 의존성을 설치합니다. 무거운 선택 의존성은 필요할 때만 추가합니다.

```sh
uv sync --extra yolo        # YOLOv7/torch 실시간 탐지 의존성
uv sync --extra audio-model # Hugging Face Transformers 오디오 분류기
uv sync --extra windows-ocr # Windows OCR/TensorFlow 계열 의존성
uv sync --extra dev         # pytest/ruff 개발 도구
```

기존 requirements 파일은 레거시 환경 재현용으로 유지합니다.

```sh
python -m pip install -r requirements.txt
python -m pip install -r AutoQA_llm/requirements.txt
python -m pip install -r QA-Auto/requirements_win.txt
python -m pip install -r QA_video_process/requirements.txt
```

로컬 환경 변수는 `.env.example`을 참고해 설정합니다.

```sh
cp .env.example .env
```

사운드 로그 추출에는 FFmpeg가 필요합니다. `ffmpeg`가 `PATH`에 없으면 아래 환경 변수 중 하나를 지정합니다.

```sh
export COMPRESSO_FFMPEG_PATH="/path/to/ffmpeg"
export FFMPEG_PATH="/path/to/ffmpeg"
```

Windows PowerShell 예시는 다음과 같습니다.

```powershell
$env:FFMPEG_PATH = "C:\tools\ffmpeg\bin\ffmpeg.exe"
```

Gemini API 분석을 사용할 때만 API 키가 필요합니다. 로컬 사운드 로그 추출 자체는 API 키 없이 동작합니다.

```sh
export GEMINI_API_KEY="your-api-key"
```

## 실행

루트 AutoQA GUI:

```sh
uv run autoqa-macro
```

YOLOv7 실시간 탐지 GUI:

```sh
uv run autoqa-yolo
```

LLM 비디오 QA 및 사운드 로그 분석 UI:

```sh
uv run autoqa-llm
```

WSL/Linux에서 Qt 플랫폼 오류가 나면 다음처럼 실행합니다.

```sh
QT_QPA_PLATFORM=wayland uv run autoqa-llm
```

## 사운드 이펙트 로그 검증 흐름

`AutoQA_llm`의 비디오 분석은 API 분석 전에 로컬 오디오 분석을 먼저 수행합니다.

1. `AutoQA_llm/main.py`를 실행합니다.
2. 게임 플레이 비디오를 선택합니다.
3. 선택 사항으로 기대 사운드 이벤트 JSON을 지정합니다.
4. 분석을 시작하면 FFmpeg로 오디오를 WAV로 추출합니다.
5. `sound_event_detector.py`가 에너지, 방향성, 시간, 프레임, 신뢰도 기반 사운드 이벤트를 생성합니다.
6. 기대 이벤트가 있으면 `missing_sound`, `desync`, `type_mismatch`, `direction_mismatch`, `unexpected_sound` 이슈를 계산합니다.
7. Gemini API 키가 있으면 로컬 사운드 로그가 비디오 분석 프롬프트에 근거 데이터로 포함됩니다.
8. API가 없거나 실패해도 로컬 사운드 로그와 검증 이슈는 버그 목록에 추가됩니다.

기대 이벤트 JSON 예시는 [docs/sound_expected_events.example.json](./docs/sound_expected_events.example.json)에 있습니다.

```json
{
  "events": [
    {
      "time": "0:01.20",
      "type": "Impact",
      "direction": "left",
      "tolerance_seconds": 0.35,
      "description": "Player collision should trigger an impact sound."
    }
  ]
}
```

## 검증

사운드 이벤트 추출 로직은 단위 테스트로 확인할 수 있습니다.

```sh
uv run autoqa-check
uv run python -m unittest AutoQA_llm.test_sound_event_detector test_logger
uv run python -m compileall -q autoqa_tools AutoQA_llm/sound_event_detector.py AutoQA_llm/test_sound_event_detector.py test_logger.py
```

현재 사운드 로그 경로는 로컬 deterministic extractor를 기본값으로 사용합니다. 더 강한 로컬 분류가 필요하면 `TransformersAudioClassifier` 어댑터를 통해 Hugging Face audio-classification 모델을 붙일 수 있습니다.

## 데모 이미지

현재 checkout에 포함된 화면 자산 기준 예시는 다음과 같습니다.

### GUI Result

![AutoQA GUI result](./screen/gui_result.jpg)

### OCR Result

![AutoQA OCR result](./QA-Auto/screen/result_ocr.png)

### Canny Result

![AutoQA Canny result](./QA-Auto/screen/result_canny.jpg)

## 문제 해결

- `git lfs` 경고가 나오면 Git LFS를 설치하고 `git lfs install && git lfs pull`을 다시 실행합니다.
- PyQt5가 디스플레이를 찾지 못하면 Windows 네이티브 Python에서 실행하거나 `QT_QPA_PLATFORM=wayland`를 지정합니다.
- `ffmpeg not found` 오류가 나오면 `FFMPEG_PATH` 또는 `COMPRESSO_FFMPEG_PATH`를 설정합니다.
- Gemini 분석이 실패해도 로컬 사운드 로그 추출은 계속 사용할 수 있습니다.
- Windows 창 제어, 마우스/키보드 자동화, 프로세스 선택 기능은 WSL보다 Windows 데스크톱 Python에서 검증하는 것이 안전합니다.

## 변경 이력

최신 작업에서는 워크숍의 Sound Effect 요구사항을 반영해 실제 비디오 오디오에서 사운드 이벤트 로그를 추출하고, 기대 이벤트 JSON과 비교하는 경로를 추가했습니다. 오래된 이력은 [CHANGELOG.md](./CHANGELOG.md)를 참고하세요.

## 관련 저장소

- QA video processing reference: <https://github.com/eglabsid/QA-video-process>
