

from functools import wraps
from PIL import Image


import numpy as np
import cv2
import os,sys

import threading

import platform

# import gc
import psutil
from memory_profiler import profile
import pygetwindow as gw

import numpy as np


import warnings
warnings.filterwarnings("ignore", message="Apply externally defined coinit_flags: 2")

current_os = platform.system()
if current_os == "Windows":
    # pyqt5 와 pywinauto 충돌문제
    # COM 라이브러리를 단일 스레드 아파트먼트(STA) 모드로 초기화
    sys.coinit_flags = 2  # STA 모드 설정
    import ctypes
    from ctypes import windll

    # DPI 인식 활성화
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    
    import win32gui
    import win32ui
    # import win32con
    import win32process
    
    from pywinauto import Application, mouse, keyboard

    import mss
    import mss.tools

def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully!")
    else:
        print(f"Directory '{dir_path}' already exists.")

def os_specific_task(os_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_os = platform.system()
            if current_os == os_name:
                return func(*args, **kwargs)
            else:
                print(f"Skipping {func.__name__} on {current_os}")
        return wrapper
    return decorator

class WindowProcessHandler():
    
    __slot__ = ['hwnd','window_process']
    
    def __init__(self):
        # # DPI 인식 활성화
        # ctypes.windll.shcore.SetProcessDpiAwareness(2)
        
        # self.frame = frame
        
        self.hwnd = None
        self.window_process = None
    
    def caputer_monitor_to_cv_img(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor) # screenshot 
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # folder_dir = os.getcwd()+"/screen"
            # create_directory_if_not_exists(folder_dir)
            # saved_file = folder_dir+"/test.jpg"
            # cv2.imwrite(f"{saved_file}",image)
            # print(f"Screenshot saved as {saved_file}")
        return image
    
    # 모든 프로세스 목록 가져오기
    def get_running_process_list(self):
        process_list = []
        for proc in psutil.process_iter(['pid', 'name']):
            process_info = proc.info
            process_list.append(process_info)
        process_list = sorted(process_list,key=lambda x:x['name'])
        return process_list

    # 특정 프로세스 찾기 (예: 'notepad.exe')
    def find_process_by_name(self, process_name):
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == process_name:
                return proc
        return None

    def connect_application_by_process_name(self,process_name):
        message = ""
        # 실행 중인 프로세스 검색
        proc = self.find_process_by_name(process_name)
        if not proc:
            message = f" '{process_name}' 해당 프로세스가 실행중이지 않습니다."
            return message
            
        if proc.info['name'] == process_name:
            message = self.check_process(proc)
            print(message)

            # self.hwnd = self.get_handler_of_window_process(proc.info['name'])
            # 윈도우 활성화
            self.hwnd, self.window_process = self.find_hwnd_window_by_pid(proc.info['pid'])
            if self.window_process:
                self.window_process.activate()
                message += "\n Window activated successfully!"
            else:
                message += "\n Window not found."
                
            print(message)
            return message

        message = f"Process '{process_name}' not found."
        print(message)
        return message

    # 프로세스 ID를 기반으로 윈도우 찾기
    @os_specific_task("Windows")
    def find_hwnd_window_by_pid(self,pid):
        def callback(hwnd, pid):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid:
                hwnds.append(hwnd)
            return True

        hwnds = []
        win32gui.EnumWindows(callback, pid)
        if hwnds:
            return hwnds[0], gw.Win32Window(hwnds[0])
        return None, None

    # 프로세스 로드 상태 체크 함수
    def check_process(self,proc):
        message = f"Process {proc.info['name']} (PID: {proc.info['pid']}) is running."
        try:
            if proc.is_running():
                cpu_usage = proc.cpu_percent(interval=1)
                memory_info = proc.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)  # 메모리 사용량 (MB)
                message = f"CPU Usage: {cpu_usage}% \n"
                message += f"Memory Usage: {memory_usage} MB"
            else:
                message = "Process is not running."
        except psutil.NoSuchProcess:
            message = "Process no longer exists."
        return message  
    
    @os_specific_task("Windows")
    def mouseclick(self, button: str, coords: tuple):
        def task():
        # self.window_process.set_focus()
            self.window_process.activate()
            mouse.click(button=button, coords=coords)
        threading.Thread(target=task).start()
    
    @os_specific_task("Windows")
    def sendkey(self, key: str):
        def task():
        # self.window_process.set_focus()
            self.window_process.activate()
            keyboard.send_keys(key)
        threading.Thread(target=task).start()
    
    @os_specific_task("Windows")
    def captuer_screen_on_application(self):
        
        try:
            if not self.hwnd:
                print(f"프로세스 '{self.process_name}'을 찾을 수 없습니다.")
                return None
            
            # Get the window rectangle
            left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
            w = right - left
            h = bot - top

            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

            saveDC.SelectObject(saveBitMap)

            # Use the default flag to capture the window
            result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0x00000001)
            # result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0)

            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)

            # Ensure the buffer size matches the expected size
            expected_size = bmpinfo['bmWidth'] * bmpinfo['bmHeight'] * 4  # BGRX format
            if len(bmpstr) != expected_size:
                # Adjust the buffer size to match the expected size
                bmpstr = bmpstr.ljust(expected_size, b'\x00')

            # Convert buffer to numpy array and reshape
            bmp_array = np.frombuffer(bmpstr, dtype=np.uint8)
            bmp_array = bmp_array.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))

            screenshot = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmp_array, 'raw', 'BGRX', 0, 1)

            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
        
            if result == 1:
                # folder_dir = os.getcwd()+"/screen"
                # create_directory_if_not_exists(folder_dir)
                # saved_file = folder_dir+"/capture.jpg"
                # screenshot.save(saved_file)
                # print(f"Screenshot saved as {saved_file}")
                return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            else:
                print("Failed to capture window")
                return None
        
        except Exception as e:
            print(f"Error: {e}")
            return None
        
@os_specific_task("Windows")
def main():
    process_name = "GeometryDash.exe"  # 예: "Notepad", "Chrome" 등
    # frame = cv2.imread('result.png',0)
    handler = WindowProcessHandler()
    handler.connect_application_by_process_name(process_name)
    # handler.captuer_screen_on_application()
    # handler.connect_application_by_handler()
    handler.window_process.activate()
    handler.mouseclick('left',(250,250))
    handler.sendkey('{SPACE}') 
    # {ENTER},{TAB},{ESC},{SPACE},{BACKSPACE} ex) 엔터 두번, {ENTER 2}
    # {DELETE},{UP},{DOWN},{LEFT},{RIGHT}
    # + (Shift), ^ (Ctrl), % (Alt)와 함께 사용 가능 (예: ^a는 Ctrl+A, +{INS}는 Shift+Insert)
    # 예를 들어, send_keys('^a^c')는 Ctrl+A로 모두 선택하고 Ctrl+C로 복사하는 동작을 수행
if __name__ == "__main__":
    main()
