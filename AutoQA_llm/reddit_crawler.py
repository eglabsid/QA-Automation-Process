import requests
import os
import time
import json
from tqdm import tqdm
from urllib.parse import urljoin
import cv2

class RedditVideoCrawler:
    def __init__(self):
        """Reddit 크롤러 초기화"""
        self.base_url = "https://www.reddit.com/r/GamePhysics"
        self.api_url = "https://www.reddit.com/r/GamePhysics.json"
        self.download_dir = "downloaded_videos"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 다운로드 디렉토리 생성
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
    
    def get_video_duration(self, video_url):
        """동영상 URL의 길이를 확인"""
        try:
            # HEAD 요청으로 동영상 크기 확인
            response = requests.head(video_url, headers=self.headers, allow_redirects=True)
            response.raise_for_status()
            
            # Content-Length 헤더에서 파일 크기 확인
            content_length = int(response.headers.get('content-length', 0))
            
            # 대략적인 동영상 길이 추정 (1MB당 약 10초로 가정)
            # 이는 매우 대략적인 추정치이며, 실제 길이와 다를 수 있습니다
            estimated_duration = content_length / (1024 * 1024) * 10
            
            return estimated_duration
        except Exception as e:
            print(f"동영상 길이 확인 중 오류 발생: {str(e)}")
            return None
    
    def get_video_url(self, post_data):
        """게시물 데이터에서 비디오 URL 추출"""
        try:
            # 1. v.redd.it 비디오 확인
            if 'secure_media' in post_data and post_data['secure_media']:
                if 'reddit_video' in post_data['secure_media']:
                    return post_data['secure_media']['reddit_video']['fallback_url']
            
            # 2. 외부 링크 확인
            url = post_data.get('url', '')
            if any(url.endswith(ext) for ext in ['.mp4', '.webm', '.gifv']):
                if url.endswith('.gifv'):
                    return url.replace('.gifv', '.mp4')
                return url
            
            return None
        except Exception as e:
            print(f"비디오 URL 추출 실패: {str(e)}")
            return None
    
    def download_video(self, url, filename):
        """비디오 다운로드"""
        try:
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(filename, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filename)) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"다운로드 실패: {url} - {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)  # 실패한 파일 삭제
            return False
    
    def crawl_videos(self, max_videos=500):
        """GamePhysics 서브레딧에서 비디오 크롤링"""
        print(f"GamePhysics 서브레딧에서 최대 {max_videos}개의 비디오를 크롤링합니다...")
        print("30초 미만의 동영상만 다운로드합니다.")
        
        downloaded_count = 0
        skipped_count = 0
        after = None
        
        while downloaded_count < max_videos:
            try:
                # API 요청 URL 구성
                params = {'after': after} if after else {}
                response = requests.get(self.api_url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                # 게시물 없으면 종료
                if 'data' not in data or 'children' not in data['data'] or not data['data']['children']:
                    print("더 이상 게시물을 찾을 수 없습니다.")
                    break
                
                # 다음 페이지 토큰 저장
                after = data['data'].get('after')
                
                # 각 게시물 처리
                for post in data['data']['children']:
                    if downloaded_count >= max_videos:
                        break
                    
                    post_data = post['data']
                    post_id = post_data['id']
                    
                    # 이미 처리된 파일인지 확인
                    filename = os.path.join(self.download_dir, f"{post_id}.mp4")
                    if os.path.exists(filename):
                        skipped_count += 1
                        continue
                    
                    # 비디오 URL 추출
                    video_url = self.get_video_url(post_data)
                    if not video_url:
                        skipped_count += 1
                        continue
                    
                    # 동영상 길이 확인
                    duration = self.get_video_duration(video_url)
                    if duration is None or duration >= 30:  # 30초 이상이면 건너뛰기
                        print(f"동영상 길이가 30초 이상으로 추정됩니다 ({duration:.1f}초). 건너뜁니다.")
                        skipped_count += 1
                        continue
                    
                    # 비디오 다운로드
                    print(f"\n게시물 처리 중: {post_data['title']}")
                    print(f"추정 동영상 길이: {duration:.1f}초")
                    if self.download_video(video_url, filename):
                        downloaded_count += 1
                        print(f"다운로드 완료: {post_data['title']}")
                    else:
                        skipped_count += 1
                    
                    # 요청 간 지연
                    time.sleep(1)
                
                if not after:
                    print("더 이상 페이지가 없습니다.")
                    break
                
            except Exception as e:
                print(f"페이지 처리 중 오류 발생: {str(e)}")
                time.sleep(5)  # 오류 발생 시 잠시 대기
                continue
        
        print(f"\n크롤링 완료!")
        print(f"총 다운로드: {downloaded_count}개")
        print(f"건너뛴 게시물: {skipped_count}개")

def main():
    # 크롤러 초기화
    crawler = RedditVideoCrawler()
    
    # 크롤링 시작
    crawler.crawl_videos(max_videos=500)

if __name__ == "__main__":
    main() 