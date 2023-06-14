# Imports
from typing import Generator
import os
import subprocess
import cv2 as cv
import numpy as np

def generate_video(folder):
    """
    Generate a video from a folder containing PNG frames.
    
    Arguments:
    folder (str): Folder path containing PNG frames
    
    The function uses FFmpeg to convert the PNG frames into a video with the 'libx264' codec and 'yuv420p' pixel format.
    The generated video is saved as 'output.mp4' in the given folder.
    """

    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', f'{folder}/%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',f'{folder}/output.mp4'
    ])

def make_dir(path):
    """
    Create a directory if it does not exist.
    
    Arguments:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.mkdir(path)

def get_video_metadata(file_path):
    """
    Retrieve metadata of a video file.
    
    Arguments:
        file_path (str): Path to the video file
    
    Returns:
        video_dict (dict): Dictionary containing video metadata (path, fps, length, height, width)
    """
    
    capture = cv.VideoCapture(file_path)
    video_dict = {
    "path": file_path,
    "fps": capture.get(cv.CAP_PROP_FPS),
    "length": int(capture.get(cv.CAP_PROP_FRAME_COUNT)),
    "height": int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
    "width": int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    }
    capture.release()
    return video_dict

def frame_generator(path:str) -> Generator[np.ndarray, None, None]:
    """
    Generate frames from a video file.
    
    Arguments:
        path (str): Path to the video file
    
    Yields:
        frame (np.ndarray): Numpy array representing a video frame
    
    The function opens the video file and yields each frame as a Numpy array.
    It automatically releases the video capture when all frames have been processed.
    """
    video = cv.VideoCapture(path)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        yield frame
    
    video.release()