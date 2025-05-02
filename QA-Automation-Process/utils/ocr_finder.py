# import easyocr
# import cv2

# # EasyOCR 리더 생성
# reader = easyocr.Reader(['en', 'ko'],gpu=True)  # 필요한 언어를 추가하세요

# # 이미지 로드
# image_path = 'screen/screenshot.jpg'
# image = cv2.imread(image_path)

# # 텍스트 인식
# results = reader.readtext(image)

# # 결과 출력
# for (bbox, text, prob) in results:
#     # bbox는 텍스트의 위치를 나타내는 네 개의 좌표입니다
#     print(f"Text: {text}, Probability: {prob}")
#     # print(f"Bounding Box: {bbox}")

#     # 텍스트 위치에 사각형 그리기
#     (top_left, top_right, bottom_right, bottom_left) = bbox
#     top_left = tuple(map(int, top_left))
#     bottom_right = tuple(map(int, bottom_right))
#     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# # 결과 이미지 저장
# cv2.imwrite('result_1.png', image)

from PyQt5.QtCore import pyqtSignal, QThread
from tqdm import tqdm

import numpy as np

import keras_ocr
import cv2
import threading

'''
 pip install tensorflow==2.13.0 keras==2.13.1 keras-ocr==0.9.3
'''

class OCRFinder(QThread):
    # finished = pyqtSignal(np.ndarray)
    finished = pyqtSignal(list)
    
    __slot__ = ["regions","lock","results","frame"]
    def __init__(self):
        super().__init__()
        self.regions = []
        # self.pipeline = keras_ocr.pipeline.Pipeline()# Keras-OCR 파이프라인 생성
        # 이미지 로드
        self.lock = threading.Lock()# 멀티쓰레드 설정
        self.results =[]
        self.frame = None
    
    def set_frame(self,frame):
        self.frame = frame
        
    def set_regions(self,image_paths):
        self.regions = [keras_ocr.tools.read(image_path) for image_path in image_paths]
    
    def set_region(self,image:np.array):
        # canny_threshold1 = 220
        # canny_threshold2 = 500
        # edges_template = cv2.Canny(image, canny_threshold1, canny_threshold2)
        # edges_template = cv2.cvtColor(edges_template, cv2.COLOR_GRAY2RGB)
        # self.regions.append(edges_template)
        self.regions.append(image)
        
        
    def recognize(self,semaphore,image,pbar):
        with semaphore:
            pbar.update(1)
            pipeline = keras_ocr.pipeline.Pipeline()# Keras-OCR 파이프라인 생성
            prediction_groups = pipeline.recognize([image])
            with self.lock:
                self.results.extend(prediction_groups[0])
                # for (text, box) in enumerate(prediction_groups[0]):
                #     (top_left, top_right, bottom_right, bottom_left) = box
                #     top_left = tuple(map(int, top_left))
                #     top_right = tuple(map(int, top_right))
                #     bottom_right = tuple(map(int, bottom_right))
                #     bottom_left = tuple(map(int, bottom_left))
                
    def run(self):
        
        self.results.clear()
        
        threads = []
        max_threads = 4
        semaphore = threading.Semaphore(max_threads)
        
        self.total_tasks = len(self.regions)
        # print(f"self.total_tasks : { self.total_tasks }")
        # 각 영역에 대해 스레드 생성 및 시작
        with tqdm(total=self.total_tasks, desc="Finding OCR") as pbar:
            for region in self.regions:
                thread = threading.Thread(target=self.recognize, args=(semaphore, region, pbar))
                threads.append(thread)
                thread.start()

            # 모든 스레드가 완료될 때까지 대기
            for thread in threads:
                thread.join()
            
            self.finished.emit(self.results)
            self.regions.clear()
    
    def stop(self):
        self.wait()
        
    def draw(self):
        # draw_results = []
        # 결과 출력 및 바운딩 박스 그리기
        for i, (text, box) in enumerate(self.results):
            # print(f"Image {i+1} - Text: {text}")
            # print(f"Image {i+1} - Text: {text}, Bounding Box: {box}")

            # 바운딩 박스 좌표 추출
            (top_left, top_right, bottom_right, bottom_left) = box
            top_left = tuple(map(int, top_left))
            top_right = tuple(map(int, top_right))
            bottom_right = tuple(map(int, bottom_right))
            bottom_left = tuple(map(int, bottom_left))

            # 바운딩 박스 그리기
            cv2.line(self.frame, top_left, top_right, (0, 255, 0), 4)
            cv2.line(self.frame, top_right, bottom_right, (0, 255, 0), 4)
            cv2.line(self.frame, bottom_right, bottom_left, (0, 255, 0), 4)
            cv2.line(self.frame, bottom_left, top_left, (0, 255, 0), 4)

            # 텍스트 표시
            cv2.putText(self.frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)

            
            # 결과 이미지 저장
            # result_path = f'result_{i+1}.png'
            # cv2.imwrite(result_path, self.frame)
            # print(f"Result saved to {result_path}")
        result_path = f'screen/result_ocr.png'
        cv2.imwrite(result_path, self.frame)
        print(f"Result saved to {result_path}")
        return self.frame