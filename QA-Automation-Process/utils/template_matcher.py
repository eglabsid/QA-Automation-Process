import cv2
import numpy as np
import threading

import matplotlib.pyplot as plt

import os
import glob
from tqdm import tqdm

from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar
from PyQt5.QtCore import pyqtSignal, QThread

from utils.score_of_sds import find_best_match,resize_image

from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from threading import Semaphore

def get_subfolders(root_folder):
    subfolders = []
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    return subfolders

def get_all_images(root_folder):
    # 이미지 파일 확장자 목록
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    # 모든 이미지 파일 경로를 저장할 리스트
    all_images = []
    
    # 하위 폴더를 포함한 모든 이미지 파일 검색
    for extension in image_extensions:
        # all_images.extend(glob.glob(os.path.join(root_folder, '**', extension), recursive=False))
        all_images.extend(glob.glob(os.path.join(root_folder, extension), recursive=False))

    return all_images

class UITemplateMatcher(QThread):
    update_progress = pyqtSignal(int, int)  # 현재 진행 상황과 총 작업 수를 전달하는 시그널
    finished = pyqtSignal(np.ndarray)  # 작업 완료 시 결과 이미지를 전달하는 시그널

    __slot__ = ['frame','templates','scale_range','threshold','lock']
    
    # def __init__(self, frame, templates, scale_range, threshold=0.8):
    def __init__(self, scale_range, threshold=0.8):
        super().__init__()
        # self.frame = frame
        # self.templates = templates
        self.frame = None
        self.templates = []
        self.gray_frame = None
        
        self.scale_range = scale_range
        self.threshold = threshold
        
        self.matches = []
        self.lock = threading.Lock()
        # self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.iter = 0
    
    def update_img_datas(self, frame, templates):
        self.frame = frame
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.templates = templates
        
    
    def match_difference_frames(self, src, des):
        # 두 프레임을 그레이스케일로 변환
        gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(des, cv2.COLOR_BGR2GRAY)

        # 두 프레임 사이의 차이 계산
        diff = cv2.absdiff(gray1, gray2)

        # 차이 이미지를 이진화
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # 변화된 부분을 강조하기 위해 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_diff = False
        threshhold = 500
        # 원본 프레임에 변화된 부분을 강조
        for contour in contours:
            if cv2.contourArea(contour) > threshhold:  # 너무 작은 변화는 무시
                is_diff = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 5)
                break

        return is_diff
    
    def sds_multi_scale_template_matching(self,semaphore, template, pbar):
        with semaphore:
            # Dict 처리
            template_tuple = [ (k,v) for k,v in template.items()][0]
            # name = template_tuple[0]
            template_img = template_tuple[1]
            
            best_match = None
            best_val = -1
            best_scale = 1.0
            best_loc = (-1, -1)
            # total_range = int(abs(self.scale_range[1]-self.scale_range[0]) // self.self.scale_range[2])
 
            for i,scale in enumerate(np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2])):
                
                resized_template = cv2.resize(template_img, (0, 0), fx=scale, fy=scale)
                window_size = resized_template.shape[:2]
                stride = 10 # 10
                best_score, best_loc, best_scale = find_best_match(resized_template,self.frame,window_size, stride,self.scale_range)
                
                pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
                self.current_task += 1
                self.update_progress.emit(self.current_task,self.total_tasks)
                with self.lock:
                    # 히트맵 생성
                    # plt.imshow(result, cmap='hot', interpolation='nearest')
                    # plt.colorbar()
                    # plt.title('Heatmap of Image')
                    # plt.show()
                    print(f"sds_multi_scale_template_matching : {best_loc}")
                    self.matches.append((best_loc, best_scale, best_score, template_tuple))
        
    def multi_scale_template_matching(self,semaphore, template, pbar):
        with semaphore:
            # Dict 처리
            template_tuple = [ (k,v) for k,v in template.items()][0]
            # name = template_tuple[0]
            template_img = template_tuple[1]
            
            best_match = None
            best_val = -1
            best_scale = 1.0
            best_loc = (-1, -1)
            is_match = False
            # total_range = int(abs(self.scale_range[1]-self.scale_range[0]) // self.scale_range[2])
            
            # Canny 엣지 검출기 임계값 설정
            # canny_threshold1 = 150
            # canny_threshold2 = 300
            
            for i,scale in enumerate(np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2])):
                resized_template = cv2.resize(template_img, (0, 0), fx=scale, fy=scale)
                
                # Canny 엣지 검출기 적용
                # edges_image = cv2.Canny(self.gray_frame, canny_threshold1, canny_threshold2)
                # edges_template = cv2.Canny(resized_template, canny_threshold1, canny_threshold2)
                # result = cv2.matchTemplate(edges_image, edges_template, cv2.TM_CCOEFF_NORMED)
                
                result = cv2.matchTemplate(self.gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
                self.current_task += 1
                self.update_progress.emit(self.current_task,self.total_tasks)
                with self.lock:
                    
                    if max_val < self.threshold:
                        continue
                    
                    if max_val > best_val:
                        best_val = max_val
                        best_match = resized_template
                        best_scale = scale
                        best_loc = max_loc
                        is_match = True
                        # tmp_scale_factor = 1//scale_factor
                        # best_loc = (int(max_loc[0]*tmp_scale_factor),int(max_loc[1]*tmp_scale_factor))
                    
                    if is_match:
                        break
            
            with self.lock:
                if not is_match: # match 결과물이 없는 경우 예외처리
                    return
                self.matches.append((best_loc, best_scale, result[best_loc[1], best_loc[0]], template_tuple))
                
    def run(self):
        # print(f"Start ! UITemplateMatcher")
        self.matches.clear()
        threads = []
        works_len = len(self.templates)
        self.total_tasks = works_len * len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]))
        self.current_task = 0

        # 최대 스레드 개수를 8으로 제한
        max_threads = 8
        semaphore = threading.Semaphore(max_threads)
        
        with tqdm(total=self.total_tasks, desc="Matching templates") as pbar:
            while len(self.templates) > 0:
                template = self.templates.pop(0)
                thread = threading.Thread(target=self.multi_scale_template_matching, args=(semaphore,template, pbar))
                # thread = threading.Thread(target=self.sds_multi_scale_template_matching, args=(semaphore,template, pbar))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            result_image = self.draw_matches(self.frame)
            self.finished.emit(result_image)
    
    def stop(self):
        # self.running = False
        self.wait()
        
    def draw_matches(self, image):
        # print(f"len self.matches : {self.matches} ,{len(self.matches)}")
        for i,(loc, scale, score, template) in enumerate(self.matches):
            print(f"{loc}")
            top_left = loc
            bottom_right = (top_left[0] + int(template[1].shape[1] * scale), top_left[1] + int(template[1].shape[0] * scale))
            
            name = template[0].split("\\")[-1]
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 4)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'{name}', (top_left[0], bottom_right[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            # cv2.imwrite(f"obs_result/{i}.jpg",image)
        return image
        
class TemplateMatcher:
    def __init__(self, template, scale_range, threshold=0.8):
        # self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.template = template
        self.scale_range = scale_range
        # template 하나에 대해서만 추출할 경우
        self.best_val = -1
        self.best_match = None
        self.best_scale = None
        self.best_loc = None
        # template 과 같은 여러 개체 추출할 경우
        self.threshold = threshold
        self.matches = []
        self.lock = threading.Lock()
        # current progress state
        self.current_task = 0
        self.total_task = 0
        self.templates = []
        
        # 실험용
        self.lab_exp_1 = []
        self.lab_exp_2 = []
        self.lab_cnt = 0

    def multi_scale_match_templates(self, semaphore, gray_frame, pbar):
        with semaphore:
            for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]):
                resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
                result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.threshold)

                pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
                self.current_task += 1
                with self.lock:
                    for loc in zip(*locations[::-1]):
                        self.matches.append((loc, scale, result[loc[1], loc[0]]))

    def a_match_template(self,semaphore, gray_frame):
        
        with semaphore:
        
            str_name = "Template Matching"
            best_match = None
            best_val = -1
            best_scale = 1.0
            best_loc = (-1, -1)
                
            # 템플릿 매칭을 수행합니다.
            result = cv2.matchTemplate(gray_frame, self.template, cv2.TM_CCOEFF_NORMED)    
            # 매칭 결과에서 최대값과 위치를 가져옵니다.
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            locations = np.where(result >= self.threshold)
            # pbar.update(1)
            with self.lock:
                for loc in zip(*locations[::-1]):
                    max_loc = loc
                    max_val = result[loc[1], loc[0]]
                    # best_scale
                    if max_val > best_val:
                        best_val = max_val
                        best_match = self.template
                        # best_scale = scale
                        best_loc = max_loc 
                    
            return best_loc, best_scale, best_val, str_name
            
                        
    def multi_scale_match_a_template(self, semaphore, gray_frame, scale):#, pbar):
    # def multi_scale_match_a_template(self, gray_frame, scale):
        with semaphore:
            
            str_name = "Mult-Scale Template Matching"
            best_match = None
            best_val = -1
            best_scale = 1.0
            best_loc = (-1, -1)
            
            resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
            result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # pbar.update(1)
            with self.lock:
                if max_val < self.threshold:
                    pass
            
                if max_val > best_val:
                    best_val = max_val
                    best_match = resized_template
                    best_scale = scale
                    best_loc = max_loc 
                    
                    
            return best_loc, best_scale, result[best_loc[1], best_loc[0]], str_name
                    
    def multi_scale_match_mixed_templates(self,semaphore, gray_frame, template_tuple, pbar):
        with semaphore:
            template = template_tuple[1]
            
            best_match = None
            best_val = -1
            best_scale = 1.0
            best_loc = (0, 0)
            
            for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]):
                resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
                # locations = np.where(result >= self.threshold)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
                self.current_task += 1
                
                with self.lock:
                    if max_val > best_val and max_val >= self.threshold:
                        best_val = max_val
                        best_match = resized_template
                        best_scale = scale
                        best_loc = max_loc
            
            with self.lock:
                self.matches.append((best_loc, best_scale, result[best_loc[1], best_loc[0]], template_tuple))
    
    # templates 는 dictionary를 갖고 있는 list 타입
    def get_mixed_multi_scale_match(self, image, templates):
        self.matches.clear()
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        threads = []
        self.total_task = len(templates) * len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]))
        self.current_task = 0
        
        max_threads = 8
        semaphore = threading.Semaphore(max_threads)
        
        with tqdm(total=self.total_task, desc="Matching templates") as pbar:
            # for template in templates:
            while len(templates) > 0:
                template = templates.pop(0)
                # Dict 처리
                template_tuple = [ (k,v) for k,v in template.items()][0]
                
                # for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]):
                thread = threading.Thread(target=self.multi_scale_match_mixed_templates, args=(semaphore, gray_frame, template_tuple, pbar))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
                
    def get_multi_scale_matches(self, image):
        self.matches.clear()
        threads = []
        total_tasks = len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]))
        
        max_threads = 8
        semaphore = threading.Semaphore(max_threads)
        
        with tqdm(total=total_tasks, desc="Matching templates") as pbar:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thread = threading.Thread(target=self.multi_scale_match_templates, args=(semaphore, gray_frame, pbar))
            threads.append(thread)
            thread.start()

            for thread in threads:
                thread.join()
            

        return self.matches
    def experience_video(self,image):
        self.matches.clear()
        
        h,w,_ = image.shape
        print(f"해상도 : {w},{h}")
        if w > 2048 and w < 2560:
            self.scale_range=(0.5, 1.2, 0.1)
        elif w < 2048:
            self.scale_range=(0.02, 0.7, 0.02)
        else:
            self.scale_range=(0.8, 3.0, 0.1)
        
        gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        best_score = 0
        best_location = None
        best_scale = 1.0
        best_name = ""
        
        max_threads = 8
        semaphore = threading.Semaphore(max_threads)
        with ThreadPoolExecutor(max_workers=8) as executor:
            lab_muti_tm = []
            
            # Multi-Scale TM
            # with tqdm(total=total_task_lab_1, desc="Multi-Scale TM") as pbar:
            for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]):
                lab_muti_tm.append(executor.submit(self.multi_scale_match_a_template,semaphore,gray_frame,scale))
            
            # 작업이 완료될 때까지 대기하고 결과를 출력합니다.
            for lab_1 in as_completed(lab_muti_tm):
                # pbar.update(1)
                location, scale, score, best_name = lab_1.result()
                if score > best_score:
                    best_score = score
                    best_location = location
                    best_scale = scale

            self.matches.append((best_location, best_scale, best_score, best_name))
            
            # return cv2.cvtColor(np.array(self.draw_matches_lab(image)), cv2.COLOR_RGB2BGR)
        return self.matches
        
    def expierience_lab(self, image):
        self.matches.clear()
        
        h,w,_ = image.shape
        print(f"해상도 : {w},{h}")
        font_size = 1.5
        if w >= 2560 and w < 2600:
            self.scale_range=(0.5, 1.2, 0.1)
        elif w < 2048:
            self.scale_range=(0.02, 0.7, 0.02)
            font_size = 0.7
        else:
            self.scale_range=(0.8, 3.0, 0.1)
            font_size = 3.0
            
        
        gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        best_score = 0
        best_location = None
        best_scale = 1.0
        best_name = ""
        
        max_threads = 8
        semaphore = threading.Semaphore(max_threads)
        with ThreadPoolExecutor(max_workers=8) as executor:
            lab_muti_tm = []
            lab_ori_tm = []
            # Orignal TM
            # with tqdm(desc="Orignal TM") as pbar:
            lab_ori_tm.append(executor.submit(self.a_match_template,semaphore, gray_frame))
            
            for lab_2 in as_completed(lab_ori_tm):
                location, _, score, name = lab_2.result()
                self.matches.append((location, _, score, name))
                
            total_task_lab_1 =int((self.scale_range[1]-self.scale_range[0])//self.scale_range[2])
            # Multi-Scale TM
            # with tqdm(total=total_task_lab_1, desc="Multi-Scale TM") as pbar:
            for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]):
                lab_muti_tm.append(executor.submit(self.multi_scale_match_a_template,semaphore,gray_frame,scale))
            
            # 작업이 완료될 때까지 대기하고 결과를 출력합니다.
            for lab_1 in as_completed(lab_muti_tm):
                # pbar.update(1)
                location, scale, score, best_name = lab_1.result()
                if score > best_score:
                    best_score = score
                    best_location = location
                    best_scale = scale

            self.matches.append((best_location, best_scale, best_score, best_name))
            
            # return cv2.cvtColor(np.array(self.draw_matches_lab(image)), cv2.COLOR_RGB2BGR)
        return self.draw_matches_lab(image, font_size)
                # return self.draw_matches_lab(image)
        
    def get_a_multi_scale_match(self, image):
        self.matches.clear()
        threads = []
        # total_tasks = len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]))
        max_threads = 8
        semaphore = threading.Semaphore(max_threads)
        
        # gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for _ in range(max_threads):
            thread = threading.Thread(target=self.multi_scale_match_a_template, args=(semaphore, image))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        
        # for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_range[2]):
        #     self.multi_scale_match_a_template(image, scale)
        # return self.draw_multi_scale_matches(image)
        
        return self.best_val, self.best_match, self.best_scale, self.best_loc
    
    
    def get_a_match(self, image):
        self.matches.clear()
        # threads = []
        
        # gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # for _ in range(2):
        #     thread = threading.Thread(target=self.a_match_template, args=(gray_frame,))
        #     threads.append(thread)
        #     thread.start()
        
        # for thread in threads:
        #     thread.join()
        self.a_match_template(image)
        return self.draw_matches(image)
        # return self.best_val, self.best_match, self.best_scale, self.best_loc
    
    def draw_matches_lab(self,image,font_size=1.0):
        
        for (loc, scale, score, name) in self.matches:
            # print(f"{name} : {loc}")
            top_left = loc
            bottom_right = (top_left[0] + int(self.template.shape[1] * scale), top_left[1] + int(self.template.shape[0] * scale))
            if name == "Template Matching":
                # print(f"{top_left+bottom_right} < {name}")
                color = (0, 0, 0)
                cv2.rectangle(image, top_left, bottom_right, color, int(4*font_size))
                cv2.putText(image, f'{name} ,{score:.2f}', (bottom_right[0], bottom_right[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, int(2*font_size))
            elif name == "Mult-Scale Template Matching":
                # print(f"{top_left+bottom_right} < {name}")
                color = (255, 255, 255)
                cv2.rectangle(image, top_left, bottom_right, color, int(4*font_size))
                cv2.putText(image, f'{name} ,{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, int(2*font_size))
        
        path = f'lab_result/{self.lab_cnt}.jpg'
        cv2.imwrite(path,image)
        self.lab_cnt+=1
        return image
    
    def draw_matches(self, image):
        print(f"draw_matches : {len(self.matches)}")
        for (loc, scale, score) in self.matches:
            top_left = loc
            bottom_right = (top_left[0] + int(self.template.shape[1] * scale), top_left[1] + int(self.template.shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 4)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image
    
    def draw_multi_scale_matches(self, image):
        print(f"draw_multi_scale_matches : {len(self.matches)}")
        for (loc, scale, score) in self.matches:
            top_left = loc
            bottom_right = (top_left[0] + int(self.template.shape[1] * scale), top_left[1] + int(self.template.shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image
    
    def draw_mixed_matches(self, image):
        for (loc, scale, score, template) in self.matches:
            top_left = loc
            bottom_right = (top_left[0] + int(template[1].shape[1] * scale), top_left[1] + int(template[1].shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 3)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'{template[0]}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image


def main():
    # 사용 예제
    template = cv2.imread('target/b1.jpg', 0)
    image = cv2.imread('screen/screenshot.jpg', 0)
    matcher = TemplateMatcher(template, scale_range=(0.5, 1.2), scale_step=0.1, threshold=0.8)
    matches = matcher.get_multi_scale_matches(image)

    # 매치 결과를 이미지에 그리기
    result_image = matcher.draw_matches(cv2.imread('screen/screenshot.jpg'))
    cv2.imwrite('result_matchs.png', result_image)

if __name__ == "__main__":
    main()