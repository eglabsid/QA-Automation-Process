
# << Vision Transformer 기반의 오픈소스 프레임워크 모델
## DETR : https://huggingface.co/facebook/detr-resnet-50
## Deformable DETR < 
## YOLOS (You Only Look One-level Series) <

# ㄴ 위의 두 모델(RT-DETR vs YOLOS) 성능 비교 논문 : https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf

'''
OpenVINO에서 사용할 수 있는 대표적인 분류 데이터셋:

- ImageNet: 대규모 이미지 데이터셋으로, 다양한 객체와 장면을 포함하고 있습니다. 많은 딥러닝 모델이 ImageNet 데이터셋으로 훈련되었습니다.
- CIFAR-10 및 CIFAR-100: 각각 10개와 100개의 클래스가 포함된 작은 이미지 데이터셋입니다. 주로 학술 연구와 교육 목적으로 사용됩니다.
- MNIST: 손으로 쓴 숫자 이미지 데이터셋으로, 숫자 인식 모델을 훈련하는 데 자주 사용됩니다.
- Pascal VOC: 객체 탐지, 분류 및 세그멘테이션 작업에 사용되는 데이터셋입니다1.
''' 

# 로드 가능한 모델
'''
# Image Classification (이미지 분류):
 - ResNet-50: 이미지 분류 작업에 널리 사용되는 모델입니다.
 - Inception v3: 복잡한 이미지 분류 작업에 적합한 모델입니다.
 - MobileNet: 경량화된 모델로, 모바일 및 임베디드 장치에서 사용하기 적합합니다.
 
# Object Detection (객체 탐지):
 - SSD (Single Shot MultiBox Detector): 실시간 객체 탐지에 적합한 모델입니다.
 - YOLO (You Only Look Once): 빠르고 정확한 객체 탐지 모델입니다.
 - Faster R-CNN: 높은 정확도를 제공하는 객체 탐지 모델입니다.

# Natural Language Processing (자연어 처리):
 - BERT (Bidirectional Encoder Representations from Transformers): 다양한 자연어 처리 작업에 사용되는 모델입니다.
 - GPT-2: 텍스트 생성 및 언어 모델링에 사용되는 모델입니다.
 - RoBERTa: BERT의 변형 모델로, 더 나은 성능을 제공합니다.
 - Semantic Segmentation (의미론적 분할):

# DeepLabv3: 이미지의 각 픽셀을 분류하는 작업에 사용됩니다.
 - FCN (Fully Convolutional Networks): 의미론적 분할 작업에 적합한 모델입니다.

# Audio Classification (오디오 분류):
 - VGGish: 오디오 신호를 분류하는 데 사용되는 모델입니다.
 - YAMNet: 다양한 오디오 이벤트를 분류하는 모델입니
'''

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt

# Load the pre-trained model and processor
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

# Load an image
# url = 'https://example.com/sample-image.jpg'  # Replace with your image URL
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open('./video/animals.jpg')

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Process the outputs
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
# Draw the results on the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.9:  # Filter out low-confidence detections
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=4)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="green", font=font)

# Save the image
output_path = r'./video/output_image.jpg'
image.save(output_path)
print(f"Result image saved to {output_path}")

# Optionally, display the image
plt.imshow(image)
plt.axis('off')
plt.show()