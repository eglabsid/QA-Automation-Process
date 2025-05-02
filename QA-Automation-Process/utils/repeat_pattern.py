

import numpy as np
import re

from PyQt5.QtCore import QThread, Qt, pyqtSignal

from enum import Enum
class ItemType(Enum):
    CLICK = 0
    TYPING = 1
    REMATCH = 2
    DELAY = 3

class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
class SendKey(StrEnum):
    ENTER = '{ENTER}'
    TAB = '{TAB}'
    ESC = '{ESC}'
    SPACE = '{SPACE}'
    BACKSPACE = '{BACKSPACE}'
    DELETE = '{DELETE}'
    UP = '{DOWN}'
    LEFT = '{RIGHT}'
    Shift = '+'
    Ctrl = '^'
    Alt = '%'


class RepeatPattern(QThread):

    subfolder = pyqtSignal(str)
    finished = pyqtSignal(str)
    action_finished = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.items = []
        self.handler = None
        # self.matcher = None
        self.delay = 0.8
        
        pattern = r"quit|back"
        none_esc_patter = r"arrow|back"
        self.compiled_pattern = re.compile(pattern)
        # pattern = re.escape(string) # 특정 문자열을 정규 표현식 패턴으로 변환
        self.compiled_esc_pattern = re.compile(none_esc_patter)
        self.actions_list = []
    
    def receive_items(self, items):
        
        # _items = sorted(items,key=lambda x:x[1][0],reverse=True)
        if len(items) == 0:
            print(f"Items 사이즈가 0 이라, 루틴을 종료합니다.")
            self.running = False
            return
        
        _items = sorted(items,key=lambda x:x[1][0],reverse=True)
        
        # 정규표현식을 이용해, 돌아가는 유아이를 맨앞으로 이동         
        quit_idx = 0
        # quit_str = ""
        for idx, item in enumerate(_items):
            matches = self.compiled_pattern.findall(item[1][0])
            if len(matches) > 0:
                # quit_str = matches[0]
                quit_idx = idx
                break
        
        # 위치  변경
        # item = _items.pop(quit_idx)
        # _items.insert(0,item)
        _items = [_items[quit_idx]] + _items[:quit_idx] + _items[quit_idx+1:] 
        # print(f"_items: {_items}")
        self.items.extend(_items)
        # print(f"self.items: {self.items}")
        self.running = True
        
    def receive_handler(self, handler):
        self.handler = handler
        
    def receive_matcher(self, matcher):
        # print("receive_matcher 실행")
        self.matcher = matcher
    
    def check_usable_esc(self,pre_frame, post_frame):
        if not self.matcher.match_difference_frames(pre_frame,post_frame):
            print(f"{SendKey.ESC.name} isn't work")
            return
        
        print(f"{SendKey.ESC.name}")
        self.handler.sendkey(SendKey.ESC.value)
    
    
    def run(self): # ctrl+esc 로 종료 메시지

        
        while self.running:
            
            if len(self.items) < 1:
                print(f"Items is empty")
                self.finished.emit("모든 동작을 실행했습니다.")
                self.action_finished.emit(self.actions_list)
                break
                    
            # for item in self.items:
            data = self.items.pop(-1)
            if not self.running:
                print("is not running")
                break
            
            if data:
                ptype = data[0]
                dinfo = data[1]
                if ptype == ItemType.CLICK: # GUI 설정에 주료 활용
                    ''' 
                        GUI 설정에 주로 활용
                        다른 게임 타겟시, 활용방법 고민예정
                    '''
                    img = dinfo[0]
                    coord = dinfo[1]
                    print(f"{ptype}, {img}")
                    self.actions_list.append(data)
                    # self.msleep(int(500*self.delay))
                    pre_frame = self.handler.caputer_monitor_to_cv_img()
                    self.handler.mouseclick('left',coord)
                    self.msleep(int(1500*self.delay))
                    post_frame = self.handler.caputer_monitor_to_cv_img()
                    search = self.compiled_esc_pattern.search(img)
                    if not search:
                        self.check_usable_esc(pre_frame,post_frame)
                    self.msleep(int(800*self.delay))
                elif ptype == ItemType.TYPING:
                    # self.handler.sendkey(SendKey.ESC.value)
                    pass
                elif ptype == ItemType.REMATCH:
                    img = dinfo[0]
                    coord = dinfo[1]
                    self.handler.mouseclick('left',coord)
                    print(f"{ptype}, {img}")
                    self.actions_list.append(data)
                    self.subfolder.emit(img)
                    break
                elif ptype == ItemType.DELAY:
                    self.delay = int(dinfo[0])
            # print(f"Items : {len(self.items)}")        
        # print(f"Stop run. Items : {len(self.items)}")

    def stop(self):
        self.running = False
        print(f"Stop repeat")
        self.wait()
