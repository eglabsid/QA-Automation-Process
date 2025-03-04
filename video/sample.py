
import streamlit as st
from yt_dlp import YoutubeDL
import os

# import yt_dlp

# url = "https://www.youtube.com/watch?v=cO0yKl_xXBQ"

# # 다운로드 옵션 설정
# ydl_opts = {
#     "outtmpl": "./video/%(title)s.%(ext)s",  # 파일 저장 위치와 이름
#     "format": "best",  # 최고 품질 선택
#     "quiet": False,  # 로그 출력
#     "nocheckcertificate": True,  # 인증서 체크 무시
# }

# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([url])

# streamlit_app.py



# Streamlit 앱
st.title("YouTube Video Downloader using yt-dlp")

# 유튜브 URL 입력
youtube_url = st.text_input("Enter YouTube URL:")

# 다운로드 버튼
if st.button("Download Video"):
    if youtube_url:
        try:
            # yt-dlp 옵션 설정
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',  # 최고 품질 비디오와 오디오 다운로드
                'outtmpl': './downloads/%(title)s.%(ext)s',  # 저장 경로 및 파일 이름 형식
                'noplaylist': True,  # 재생 목록 대신 단일 동영상만 다운로드
            }

            # yt-dlp로 다운로드
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=True)
                video_title = info_dict.get('title', 'Unknown Title')
                file_path = ydl.prepare_filename(info_dict)

            # 다운로드 성공 메시지
            st.success("Video downloaded successfully!")
            st.write(f"**Video Title:** {video_title}")
            st.write(f"**Saved to:** {file_path}")

            # 파일 제공
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download File",
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime="video/mp4"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid YouTube URL.")