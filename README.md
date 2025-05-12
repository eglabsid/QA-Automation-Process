<<<<<<< Updated upstream
# AutonomousQA Tool
#### ㄴ 테스트 게임 : Geometry Dash
해당 Tool은 이미지를 기반으로 게임 Playing 이 가능한 QA Tool 입니다.

### AutoQA Macro 기능 부분 동작 내용
![AutoQA Tool Macro](./images/AutoQA_Macro.gif)

### AutoQA OCR 기능 부분 동작 내용
![AutoQA Tool OCR](./images/AutoQA_OCR.gif)

### AUTOQA VIDEO 객체 탐지 기능

👉 https://github.com/eglabsid/QA-video-process 


> [!Note]
> AutonomousQA Tool GUI
> 
![AutoQA UI](./images/tool_ui.JPG)

> [!Note] 
> 이미지를 지정할 때는 어떠한 규칙성이 필요한데, 두가지로 나누어서 설명합니다.
> 1. GUI 기능에 맞는 네이밍 규칙
> 2. 기능에 따라 진행되는 GUI가 있다면, 새로운 GUI 폴더의 네이밍은 기능명을 그대로 적용
> 3. DETR(Detection Transformer) 에 player 캐릭터를 fine-tune
> - ( ex) into_scene.jpg -> into_scene/ ) 

## Trying to Finetuning DETR

To train Dataset is made by video from Youtube (https://www.youtube.com/watch?v=cO0yKl_xXBQ) 

To

![DETR player](./images/output_detr_1.png)

![DETR player](./images/output_detr_2.png)

From

![Pretrained DETR](./images/output_image.jpg)

## 개발환경

-Language : python 3.15.5

-GUI : Pyqt5


## Install

~~~sh
pip install -r requirements_win.txt
~~~





=======
version https://git-lfs.github.com/spec/v1
oid sha256:a5d2dc9d84419e30198f8cdb0560822ffa7815cd6ac55120622facea9a944158
size 6645
>>>>>>> Stashed changes
