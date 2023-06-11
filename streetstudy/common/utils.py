import cv2 as cv
import os
import subprocess
from typing import Generator
import numpy as np

def generate_video(folder):
    """
    Given a folder, generate video from all png
    """
    subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', f'{folder}/%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    f'{folder}/output.mp4'
    ])
    return

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_video_metadata(file_path):
    capture = cv.VideoCapture(file_path)
    video_dict = {}
    video_dict["path"] = file_path
    video_dict["fps"] = capture.get(cv.CAP_PROP_FPS)
    video_dict["length"] = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    video_dict["height"] = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_dict["width"] = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    capture.release()
    return video_dict

def frame_generator(path:str) -> Generator[np.ndarray, None, None]:
    video = cv.VideoCapture(path)
    while video.isOpened():
        success, frame = video.read()
        
        if not success:
            break

        yield frame
    
    video.release()