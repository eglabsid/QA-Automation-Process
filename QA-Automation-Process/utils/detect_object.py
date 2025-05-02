
import torch
import numpy as np
import os,shutil
import cv2


# pip install torch torchvision pyautogui opencv-python pillow pywin32
#
# git clone https://github.com/ultralytics/yolov5
# cd yolov5
# pip install -r requirements.txt

folder_dir = os.getcwd()+"/screen"

class DetectObject():
    
    __slot__ = ['model','cur_screenshot','detections']
    
    def __init__(self):
        # self.model = None
        # self.load_yolo_model() # path = f"\model\yolov5s.pt"
        self.cur_screenshot = None
        self.detections = {}
    
    def load_yolo_model(self, repo_or_dir = 'ultralytics/yolov5', model ='yolov5s'):
        
        model = torch.hub.load(repo_or_dir, model)
        model.eval()
        self.model = model
        

    # # YOLOv5s 모델이 감지할 수 있는 클래스 목록
    # classes = [
    #     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    #     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    #     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    #     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    #     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    #     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    #     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    #     "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    #     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    #     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    # ]
    def detect_class_in_bounding_box(self, target_class):
        # 해당 기능은 Yolov5의 class 타입중 하나만 탐지 가능
        target_detections = self.detections[self.detections['name'] == target_class]
        
        if target_detections.empty:
            print(f"'{target_class}' 클래스의 객체를 찾을 수 없습니다.")
            return None
        
        best_detection = target_detections.iloc[0]
        x1, y1, x2, y2 = best_detection[['xmin', 'ymin', 'xmax', 'ymax']]
        
        result ={}
        result['x'] = int(x1)
        result['y'] = int(y1)
        result['width'] = int(x2) - int(x1)
        result['height'] = int(y2) - int(y1)
        
        if result:
            print(f"검출된 객체: {target_class}")
            print(f"위치: ({result['x']}, {result['y']})")
            print(f"크기: {result['width']}x{result['height']}")
        else:
            print("객체를 찾을 수 없습니다.")
        
        return result

    def detect_bounding_box_in_gui(self, screenshot):
        results = self.model(screenshot)
        
        # 결과 출력    
        self.detections = results.pandas().xyxy[0]
        for _, row in self.detections.iterrows():
            print(f"객체: {row['name']}")
            print(f"신뢰도: {row['confidence']:.2f}")
            print(f"바운딩 박스: ({row['xmin']:.0f}, {row['ymin']:.0f}, {row['xmax']:.0f}, {row['ymax']:.0f})")
            print("---")
        
        # 결과 저장 디렉토리
        result_dir = folder_dir+'/bounding_box'

        # 기존 디렉토리 삭제 및 재생성
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)

        results.save(save_dir=result_dir)
        
        return results

    def find_similar_regions(self, src_img, des_img):
        # 이미지 읽기
        img1 = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE) # IMREAD_COLOR
        img2 = cv2.imread(des_img, cv2.IMREAD_GRAYSCALE)

        # ORB 특징 검출기 생성
        orb = cv2.ORB_create()

        # 특징점과 디스크립터 계산
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # BFMatcher 생성
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 매칭 계산
        matches = bf.match(des1, des2)

        # 매칭 결과를 거리 순으로 정렬
        matches = sorted(matches, key=lambda x: x.distance)

        # 매칭 결과를 이미지에 그리기
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        
        # 결과 이미지 저장
        # folder_dir = os.getcwd()+"/screen"
        common.create_directory_if_not_exists(folder_dir)
        matches_file = folder_dir+"/similar_region.jpg"
        cv2.imwrite(matches_file, img_matches)

    def multi_scale_template_matching(self, screen_img, target_img, scale_range=(0.5, 1.5), scale_step=0.1):
        # 전체 이미지와 크롭된 이미지 읽기
        full_image = cv2.imread(screen_img, cv2.IMREAD_COLOR)
        cropped_image = cv2.imread(target_img, cv2.IMREAD_COLOR)

        best_match = None
        best_val = -1
        best_scale = 1.0
        best_loc = (0, 0)

        try:
            for scale in np.arange(scale_range[0], scale_range[1], scale_step):
                # 크롭된 이미지의 크기 조정
                resized_template = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
                result = cv2.matchTemplate(full_image, resized_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_match = resized_template
                    best_scale = scale
                    best_loc = max_loc
        except:
            pass

        top_left = best_loc
        h, w, _ = best_match.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        center_middle = (top_left[0] + w*0.5, top_left[1] + h*0.5)
        
        # 결과를 이미지에 그리기
        cv2.rectangle(full_image, top_left, bottom_right, (0, 255, 0), 2)

        # 결과 이미지 저장
        # folder_dir = os.getcwd()+"/screen"
        common.create_directory_if_not_exists(folder_dir)
        result_dir = folder_dir+"/matching.jpg"
        
        cv2.imwrite(result_dir, full_image)

        return center_middle, best_scale
        

def main():
    process_name = "Geometry Dash"  # 예: "Notepad", "Chrome" 등
    target_class = "clock"  # 예: "person", "car", "laptop" 등
    
    source_img = os.getcwd()+'/src_backup/sadCat.png'
    target_img = os.getcwd()+'/src_backup/test.png'
    
    observer = DetectObject()
    observer.find_similar_regions(source_img,target_img)
    center_pos,scale = observer.multi_scale_template_matching(source_img,target_img)
    print(center_pos)
    # Yolo section
    observer.load_yolo_model()
    results = observer.detect_bounding_box_in_gui(source_img)
    # result = observer.detect_class_in_bounding_box(target_class)
    
    # results.show()        
    
if __name__ == "__main__":
    main()
    
    



