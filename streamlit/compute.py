import os
import tempfile

import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st

import postprocess
from streetstudy.model import yolo
from streetstudy.common import utils

def video_metadata(uploaded_file):
    # Assert and read video
    if uploaded_file is None or uploaded_file.name[-3:] != "mp4":
        st.error("Please upload a valid .mp4 file")
        return RuntimeError()    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    # Video metadata capture
    video_dict = {}
    capture = cv.VideoCapture(tfile.name)
    video_dict["path"] = tfile.name
    video_dict["fps"] = capture.get(cv.CAP_PROP_FPS)
    video_dict["length"] = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    video_dict["height"] = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_dict["width"] = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    capture.release()
    return video_dict

def predict(video_dict):    
    preds = yolo.st_predict(video_dict)
    return preds

def postprocess_videos(video_dict, preds, save_path, SKIP=50):
    postprocess_list = ["heatmap", "bounding_boxes", "directional_arrows"]
    for path in postprocess_list:
        utils.make_dir(os.path.join(save_path, path))

    loading_bar_text = "Postprocessing in progress. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    valid_frames = [x for x in range(0, video_dict['length'], SKIP)]
    
    capture = cv.VideoCapture(video_dict['path'])
    while True:
        current_frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        success, frame = capture.read()
        if not success:
            break
        if current_frame_number not in valid_frames:
            continue

        for item in postprocess_list:
            fig, ax = plt.subplots()
            ax.set(xlim=(0, video_dict['width']), ylim=(video_dict['height'], 0))    
            if item == "heatmap":
                postprocess.heatmap(ax, preds, current_frame_number)
            if item == "bounding_boxes":
                postprocess.bounding_boxes(ax, preds, current_frame_number)
            if item == "directional_arrows":
                postprocess.directional_arrows(ax, preds, current_frame_number)
            ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            fig.savefig(os.path.join(save_path, video_dict[''], str(current_frame_number)))
            plt.close()
            
        loading_bar.progress(current_frame_number/video_dict["length"], text=loading_bar_text)
    
    for folder in os.listdir(save_path):
        utils.generate_video(os.path.join(save_path, folder))
    
    loading_bar.empty()
    capture.release()
    
    
