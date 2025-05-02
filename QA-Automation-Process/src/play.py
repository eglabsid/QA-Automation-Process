import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

# opencv
import cv2


import numpy as np
import mss
import mss.tools

# import gc
from memory_profiler import profile

import pygetwindow as gw

# pyautogui
import pyautogui
import subprocess, psutil

from utils.template_matcher import TemplateMatcher

window_ui = 'play.ui'

crop_size = {}
crop_size['width'] = 700
crop_size['height'] = 300

class ScreenCaptureThread(QThread):
    
    frame_captured = pyqtSignal(np.ndarray, tuple, np.ndarray)

    def __init__(self, target_img, fps=30):
        super().__init__()
        self.fps = fps
        self.running = True
        
        # 템플릿 이미지 로드
        self.template = cv2.imread(target_img, 0)
        self.template_w, self.template_h = self.template.shape[::-1]
            
    def detect_template_in_frame(self,frame,template, threshold = 0.45):
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use template matching to find the object in the frame
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(max_val)
        # Check if the match is good enough
        if max_val > threshold:  # You can adjust this threshold
            return True
        else:
            return False

    # 초기 검색을 위해 사용
    def detect_template_using_cv(self,frame, scale_range=(0.45, 0.85), scale_step=0.1):
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matcher = TemplateMatcher(self.template, scale_range, scale_step)
        # best_val, best_match, best_scale, best_loc = matcher.get_a_match(gray_frame)
        return matcher.get_a_multi_scale_match(gray_frame)
        
    
    def crop_template(self, frame, best_match, best_loc):
        
        top_left = best_loc
        # h, w, _ = best_match.shape
        h, w = best_match.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # bounding_box = [top_left,bottom_right]
        
        # Match 된 Bounding Box
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        
        center_middle = (top_left[0] + w*0.5, top_left[1] + h*0.5)
        
        # 크롭할 영역 정의 (x, y, width, height)
        w = crop_size['width']
        h = crop_size['height']
        crop_top_left = (top_left[0] - int(w*0.2), top_left[1] - int(h*0.2))
        crop_bottom_right = (crop_top_left[0] + int(w*2), crop_top_left[1] + int(h*0.7))
        # crop_w, crop_h = 250, 250
        crop_sx, crop_sy = crop_top_left[0],crop_top_left[1] 
        crop_ex, crop_ey = crop_bottom_right[0],crop_bottom_right[1] 
        cropped_img = frame[crop_sy:crop_ey, crop_sx:crop_ex]

        # 결과를 이미지에 그리기
        cv2.rectangle(frame, crop_top_left, crop_bottom_right, (255, 0, 0), 4)
        
        return center_middle, cropped_img    
    
    def run(self):
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while self.running:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                best_val, best_match, best_scale, best_loc = self.detect_template_using_cv(img)
                if self.detect_template_in_frame(img,best_match):
                    
                    center_middle,crop_img = self.crop_template(img,best_match, best_loc)

                    # self.frame_captured.emit(img,center_middle,cropped_image)
                    self.frame_captured.emit(img,center_middle,crop_img)
                    self.msleep(int(1000 / self.fps))
                

    def stop(self):
        self.running = False
        self.wait()


class playWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(playWindow, self).__init__()

        # UI 파일 로드
        uic.loadUi(window_ui, self)
        
        self.centralWidget.setLayout(self.main_layout)
        
        self.setWindowTitle("Image based Playing Tool (ver.Dev)")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.setGeometry(300, 300, 800, 900)
        
        self.log.setReadOnly(True)
        # buttons = "self.pushButton_"
        # for n in range(4):
        #     name = buttons+str(n)
        #     button = eval(name)
        #     button.setFixedHeight(80)
        #     if n == 0:
        #         button.clicked.connect(self.capture_screen)

        # fps 설정
        template_path = 'screen/target/b1.jpg'
        self.capture_thread = ScreenCaptureThread(target_img=template_path,fps=60)
        self.capture_thread.frame_captured.connect(self.update_frame)
        self.capture_thread.start()


    def get_widgets_img(self,frame, re_width, re_height):
        # PyQt5 프레임에 표시
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        frame_bytes = frame.tobytes()
        q_img = QImage(frame_bytes, width, height, bytes_per_line, QImage.Format_RGB888)

        # QImage를 리사이즈
        q_img_resized = q_img.scaled(re_width,re_height)
        return QPixmap.fromImage(q_img_resized)

    
    # @profile
    def update_frame(self, frame, pos, crop_frame):

        # pix_map = self.get_widgets_img(frame, 500, 500)
        # self.screen.setPixmap(pix_map)
        
        pix_map = self.get_widgets_img(crop_frame, crop_size['width'], crop_size['height'])
        self.monitor.setPixmap(pix_map)
                              

        self.log.append("추적하는 객체 위치 : {0},{1}".format(pos[0],pos[1]))
        

    def connect_application_by_process_name(self, process_name):
        # 메모장 실행
        process = subprocess.Popen([process_name])
        # psutil을 사용하여 프로세스 정보 가져오기
        proc = psutil.Process(process.pid)

        while not proc.is_running():
            self.check_process_load(proc)

        # 윈도우 활성화
        window = self.find_window_by_pid(proc.pid)
        if window:
            window.activate()
            print("Window activated successfully!")
        else:
            print("Window not found.")        

        
    def mouseclick(self,button:str,coords:tuple):
        # 마우스 이동
        pyautogui.moveTo(coords[0],coords[1], duration=1)
        # 마우스 클릭
        pyautogui.click()
    
    def sendkey(self,key:str):        
        pyautogui.press(key)

    # 프로세스 ID를 기반으로 윈도우 찾기
    def find_window_by_pid(self, pid):
        for window in gw.getWindowsWithTitle(''):
            if window._hWnd == pid:
                return window
        return None
    
    # 프로세스 로드 상태 체크 함수
    def check_process_load(self, proc):
        try:
            if proc.is_running():
                cpu_usage = proc.cpu_percent(interval=1)
                memory_info = proc.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)  # 메모리 사용량 (MB)
                print(f"CPU Usage: {cpu_usage}%")
                print(f"Memory Usage: {memory_usage} MB")
            else:
                print("Process is not running.")
        except psutil.NoSuchProcess:
            print("Process no longer exists.")

    def closeEvent(self, event):
        self.capture_thread.stop()
        event.accept()
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = playWindow()
    window.show()
    sys.exit(app.exec_())
