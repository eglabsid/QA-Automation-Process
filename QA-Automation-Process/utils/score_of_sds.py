import numpy as np
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from threading import Semaphore

from memory_profiler import profile
from tqdm import tqdm
import time

import cv2

# 두 이미지 로드 및 리사이즈
def load_and_resize_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image

def resize_image(image, scale_factor):
    """
    이미지를 주어진 비율로 리사이즈합니다.
    
    :param image: 입력 이미지 (NumPy 배열)
    :param scale_factor: 리사이즈 비율 (0.0 < scale_factor <= 1.0)
    :return: 리사이즈된 이미지 (NumPy 배열)
    """
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    print(f"new_dimensions : {new_dimensions}, scale_factor : {scale_factor}")
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def execution_time_decorator(func):
    """
    함수의 실행 시간을 측정하는 데코레이터.
    
    :param func: 실행할 함수
    :return: 데코레이터된 함수
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 실행 시간: {execution_time:.6f} 초")
        return result
    return wrapper

# Scalable Diversity Similarity
def sds(template, candidate, lambda_spatial=0.5, lambda_scale=0.1):
    """
    Scalable Diversity Similarity (SDS) 계산
    
    :param template: 템플릿 이미지의 특징점 집합 (N x D 배열)
    :param candidate: 후보 이미지의 특징점 집합 (M x D 배열)
    :param lambda_spatial: 공간 거리에 대한 가중치
    :param lambda_scale: 스케일 변화에 대한 페널티 가중치
    :return: SDS 점수
    """
    
    # 양방향 다양성 계산
    div_t2c = diversity(template, candidate, lambda_spatial)
    div_c2t = diversity(candidate, template, lambda_spatial)
    
    # 스케일 변화에 대한 페널티 계산
    scale_penalty = np.abs(np.log(len(template) / len(candidate)))
    
    # SDS 점수 계산
    sds_score = (div_t2c + div_c2t) / 2 - lambda_scale * scale_penalty
    
    return sds_score

def diversity(set1, set2, lambda_spatial):
    """
    한 방향의 다양성 계산
    
    :param set1: 첫 번째 특징점 집합
    :param set2: 두 번째 특징점 집합
    :param lambda_spatial: 공간 거리에 대한 가중치
    :return: 다양성 점수
    """
    
    # 외관 거리와 공간 거리 계산
    dist_appearance = cdist(set1[:, :-2], set2[:, :-2], metric='euclidean')
    dist_spatial = cdist(set1[:, -2:], set2[:, -2:], metric='euclidean')
    
    # 결합된 거리 계산
    dist_combined = dist_appearance + lambda_spatial * dist_spatial
    
    # 최근접 이웃 찾기
    nn_indices = np.argmin(dist_combined, axis=1)
    
    # 지역 순위 정보 계산
    local_ranks = calculate_local_ranks(set1, set2, nn_indices)
    
    # 다양성 점수 계산
    diversity_score = np.sum(local_ranks) / len(set1)
    
    return diversity_score

def calculate_local_ranks(set1, set2, nn_indices):
    """
    지역 순위 정보 계산
    
    :param set1: 첫 번째 특징점 집합
    :param set2: 두 번째 특징점 집합
    :param nn_indices: 최근접 이웃의 인덱스
    :return: 지역 순위 배열
    """
    
    local_ranks = np.zeros(len(set1))
    
    for i, nn_idx in enumerate(nn_indices):
        # 극좌표계 변환
        center = set2[nn_idx, -2:]
        relative_coords = set2[:, -2:] - center
        angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
        
        # 각도에 따라 정렬
        sorted_indices = np.argsort(angles)
        
        # 지역 순위 계산
        local_rank = np.where(sorted_indices == nn_idx)[0][0]
        local_ranks[i] = local_rank
    
    return local_ranks


# 기존의 sds, diversity, calculate_local_ranks 함수는 그대로 유지합니다.

def extract_features(image):
    """
    이미지에서 특징점을 추출합니다.
    
    :param image: 입력 이미지 (높이 x 너비 x 채널)
    :return: 특징점 벡터 (1 x D 배열)
    """
    # 이미지의 각 채널에 대해 히스토그램을 계산합니다.
    hist_r = np.histogram(image[:, :, 0], bins=256, range=(0, 256))[0]
    hist_g = np.histogram(image[:, :, 1], bins=256, range=(0, 256))[0]
    hist_b = np.histogram(image[:, :, 2], bins=256, range=(0, 256))[0]
    
    # 히스토그램을 하나의 벡터로 결합합니다.
    # features = np.concatenate([hist_r, hist_g, hist_b])
    # 히스토그램을 2차원 배열로 결합합니다.
    features = np.vstack([hist_r, hist_g, hist_b])
    
    # 특징점 벡터를 정규화합니다.
    features = features / np.linalg.norm(features)
    
    return features

def find_best_match_old(template, target, window_size, stride, scale_range=(0.5, 2.0, 0.1)):
    """
    주어진 템플릿에 대해 타겟 이미지에서 가장 높은 SDS 값을 가진 영역을 찾습니다.
    
    :param template: 템플릿 이미지의 특징점 집합 (N x D 배열)
    :param target: 타겟 이미지의 특징점 집합 (M x D 배열)
    :param window_size: 슬라이딩 윈도우의 크기 (높이, 너비)
    :param stride: 슬라이딩 윈도우의 이동 간격
    :param scale_range: 스케일 범위 (최소, 최대, 간격)
    :return: 최고 SDS 값, 최고 점수의 좌표 (x, y), 최고 점수의 스케일
    """
    best_score = -np.inf
    best_location = None
    best_scale = None

    height, width = target.shape[:2]
    
    for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]):
        scaled_window = (int(window_size[0] * scale), int(window_size[1] * scale))
        
        for y in range(0, height - scaled_window[0] + 1, stride):
            for x in range(0, width - scaled_window[1] + 1, stride):
                # 현재 윈도우에서 특징점 추출
                window = target[y:y+scaled_window[0], x:x+scaled_window[1]]
                window_features = extract_features(window)  # 이 함수는 별도로 구현해야 합니다
                
                # SDS 계산
                score = sds(template, window_features)
                
                if score > best_score:
                    best_score = score
                    best_location = (x, y)
                    best_scale = scale
    
    return best_score, best_location, best_scale

def sliding_window(image, step_size, window_size):
    # 슬라이딩 윈도우 생성
    for y in range(0, image.shape[0] - window_size[1]+1, step_size):
        for x in range(0, image.shape[1] - window_size[0]+1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
            
@execution_time_decorator
def find_best_match(template, target, window_size, stride, scale_range=(0.5, 2.0, 0.1)):
    """
    주어진 템플릿에 대해 타겟 이미지에서 가장 높은 SDS 값을 가진 영역을 찾습니다.
    
    :param template: 템플릿 이미지의 특징점 집합 (N x D 배열)
    :param target: 타겟 이미지의 특징점 집합 (M x D 배열)
    :param window_size: 슬라이딩 윈도우의 크기 (높이, 너비)
    :param stride: 슬라이딩 윈도우의 이동 간격
    :param scale_range: 스케일 범위 (최소, 최대, 간격)
    :return: 최고 SDS 값, 최고 점수의 좌표 (x, y), 최고 점수의 스케일
    """
    best_score = -np.inf
    best_location = None
    best_scale = None

    # scale_factor = 0.5
    # template = resize_image(template,scale_factor)
    # target = resize_image(target,scale_factor)
    
    height, width = target.shape[:2]
    semaphore = Semaphore(4)
    # def process_scale(semaphore,x, y,window, scale,pbar):
    def process_scale(semaphore,scale,pbar):
        with semaphore:
            local_score = -np.inf
            local_location = None
            local_scale = None
            
            template_features = extract_features(template)
            scaled_window = (int(window_size[0] * scale), int(window_size[1] * scale))
            for (x, y, window) in sliding_window(target, stride, scaled_window):
                pbar.update(1)
                window = target[y:y+scaled_window[0], x:x+scaled_window[1]]
                window_features = extract_features(window)
                score = sds(template_features, window_features)
                if score > local_score:
                    local_score = score
                    local_location = (x,y)
                    local_scale = scale
                
                # if score > threshold:
                #     local_score = score
                #     local_location = (x,y)
                #     local_scale = scale
                #     break
            
            return local_score, local_location, local_scale
        # pbar.update(1)
        # window_features = extract_features(window)
        # score = sds(template, window_features)
        
        # return score, (x,y), scale

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        total_tasks = sum((height - int(window_size[0] * scale) + 1) // stride * (width - int(window_size[1] * scale) + 1) // stride for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]))
        with tqdm(total=total_tasks, desc="Processing Scale") as pbar:
            for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]):
                # scaled_window = (int(window_size[0] * scale), int(window_size[1] * scale))
                # for (x, y, window) in sliding_window(target, stride, scaled_window):
                    # futures.append(executor.submit(process_scale,semaphore, x, y, window, scale, pbar))
                futures.append(executor.submit(process_scale,semaphore,scale,pbar))
        
            # 작업이 완료될 때까지 대기하고 결과를 출력합니다.
            for future in as_completed(futures):
                # pbar.update(1)
                score, location, scale = future.result()
                if score > best_score:
                    best_score = score
                    best_location = location
                    best_scale = scale
                    # resize
                    # best_location = (location[0]*1//scale_factor, location[1]*1//scale_factor)
    
    return best_score, best_location, best_scale


def main():
    # 사용 예시
    scale_factor = 0.5
    template_image = resize_image(cv2.imread('screen/UI/select_options/create.jpg', cv2.IMREAD_COLOR),scale_factor)
    target_image = resize_image(cv2.imread('screen/capture.jpg', cv2.IMREAD_COLOR),scale_factor)
    # template_image = np.array(cv2.imread('screen/UI/select_options/create.jpg', cv2.IMREAD_COLOR))
    # target_image = np.array(cv2.imread('screen/capture.jpg', cv2.IMREAD_COLOR))
    # template_image = np.random.rand(220, 340, 3)  # 가상의 템플릿 이미지
    # target_image = np.random.rand(500, 500, 3)  # 가상의 타겟 이미지

    # template_features = extract_features(template_image)
    window_size = template_image.shape[:2]
    stride = 10 # 10

    best_score, best_location, best_scale = find_best_match(template_image, target_image, window_size, stride)

    print(f"최고 SDS 점수: {best_score}")
    print(f"최고 점수의 좌표: {best_location}")
    print(f"최고 점수의 좌표: {(best_location[0]*1//scale_factor,best_location[1]*1//scale_factor)}")
    print(f"최고 점수의 스케일: {best_scale}")
    
if __name__ == "__main__":
    main()
    




