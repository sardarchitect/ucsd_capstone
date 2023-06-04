import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import cv2 as cv
import matplotlib.pyplot as plt
import torch 
import seaborn as sns
import sys
import glob
import subprocess
import os
import matplotlib.patches
import inference
from streetstudy.model import yolo
from streetstudy.common import display
from sort.sort import Sort

def generate_video(folder):
    subprocess.call([
    'ffmpeg', '-framerate', '8', '-pattern_type', 'glob', '-i', f'{folder}/*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    f'{folder}/output.mp4'
    ])
    return

def postprocess(postprocess_type, video_dict, current_frame_number, frame, predictions, save_path):
        #BOUNDING BOXES
        fig, ax = plt.subplots(figsize=(17,10))
        ax.set(xlim=(0, video_dict['width']), ylim=(video_dict['height'], 0))
        
        curr_save_path = os.path.join(save_path, postprocess_type)
        if not os.path.exists(curr_save_path):
            os.mkdir(curr_save_path)

        if postprocess_type == "bbox":
            for bbox in predictions:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (bbox[0], bbox[1]), 
                        bbox[2] - bbox[0], 
                        bbox[3] - bbox[1], 
                        rotation_point='xy',
                        facecolor='none', 
                        ec='r', 
                        lw=1)
                )
                plt.text(bbox[0], bbox[1], bbox[4])
        
        if postprocess_type == "heatmap":
            feet_x = predictions[0] + ((predictions[3] - predictions[0]) / 2)
            feet_y = predictions[1] + (predictions[4] - predictions[1])
            sns.kdeplot(x=feet_x, y=feet_y, thresh=0, levels=50, alpha=0.2, fill=True, ax=ax, cmap='hot')

        ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        fig.savefig(os.path.join(curr_save_path, str(current_frame_number)))
        plt.close()

def pipeline(uploaded_file, skip=50, save_path='ouputs/'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Loading bar
    loading_bar_text = "Inference in progress. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    
    # Assert and read video
    if uploaded_file is None or uploaded_file.name[-3:] != "mp4":
        st.error("Please upload a valid .mp4 file")
        return 0
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Init model
    model = yolo.yolov5()
    
    # Video data capture
    capture = cv.VideoCapture(tfile.name)

    # Video meta data
    video_dict = {}
    video_dict["fps"] = capture.get(cv.CAP_PROP_FPS)
    video_dict["length"] = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    video_dict["height"] = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_dict["width"] = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))

    # Init tracker
    sort = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    valid_frames = [int(x) for x in range(0, video_dict['length'], skip)]
    # Init video loop
    while True:
        current_frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        success, frame = capture.read()
        if not success:
            break
        if current_frame_number not in valid_frames:
            continue
        # Predict
        predictions = model(frame).xyxy[0]
        tracked_predictions = sort.update(predictions[:, :4].cpu().numpy())
        postprocess("bbox", video_dict, current_frame_number, frame, tracked_predictions)
        postprocess("heatmap", video_dict, current_frame_number, frame, tracked_predictions)
        
        for folder in os.listdir(save_path):
            generate_video(os.path.join(save_path, folder))
        
        loading_bar.progress(current_frame_number/video_dict["length"], text=loading_bar_text)
    
    loading_bar.empty()
    capture.release()
    return