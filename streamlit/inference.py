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


def generate_video(folder):
    subprocess.call([
    'ffmpeg', '-framerate', '8', '-pattern_type', 'glob', '-i', f'{folder}/*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    f'{folder}/out.mp4'
    ])
    return 1

def predict(video_path, model):
    loading_bar_text = "Inference in progress. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    capture = cv.VideoCapture(video_path)
    total_frame_number = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    annotations = torch.empty((0, 7), device='cuda')
    
    for current_frame_number in range(total_frame_number):
        success, frame = capture.read()
        if not success:
            break
        results = model(frame)
        current_frame_number_col = torch.ones((results.xywh[0].shape[0]), device='cuda') * current_frame_number
        list_of_preds = torch.column_stack((current_frame_number_col, results.xywh[0]))
        annotations = torch.row_stack((annotations, list_of_preds))
        loading_bar.progress(current_frame_number/total_frame_number, text=loading_bar_text)
    loading_bar.empty()
    capture.release()
    annotations_df = pd.DataFrame(data=annotations.cpu().numpy(), columns=['current_frame', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height', 'conf', 'class'])
    return annotations_df

def postprocess(video_path, annotations_df, save_path):
    loading_bar_text = "Post-processing in progress. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    capture = cv.VideoCapture(video_path)
    total_frame_number = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    image_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    image_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    valid_frames = [int(x) for x in range(0, total_frame_number, 50)]
    print(valid_frames)

    while True:
        success, frame = capture.read()
        current_frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        loading_bar.progress(current_frame_number/total_frame_number, text=loading_bar_text)
        
        if not success:
            break
        if current_frame_number not in valid_frames:
            continue

        fig, ax = plt.subplots(figsize=(10,7))
        ax.set(xlim=(0, image_width), ylim=(image_height, 0))
        current_annotations = annotations_df[annotations_df['current_frame'] == current_frame_number]
        
        #BOUNDING BOXES
        curr_save_path = os.path.join(save_path, "bbox")
        if not os.path.exists(curr_save_path):
            os.mkdir(curr_save_path)
        for _, bbox in current_annotations.iterrows():
            ax.add_patch(
            matplotlib.patches.Rectangle(
                ((bbox['bbox_center_x'] - (bbox['bbox_width'] / 2)), (bbox['bbox_center_y'] - (bbox['bbox_height'] / 2))), 
                bbox['bbox_width'], 
                bbox['bbox_height'], 
                rotation_point='xy',
                facecolor='none', 
                ec='r', 
                lw=1)
            )
        ax.imshow(frame)
        fig.savefig(os.path.join(curr_save_path, str(current_frame_number)))
        plt.close()
        generate_video(curr_save_path)
    capture.release()

def pipeline(uploaded_file):
    if uploaded_file is None or uploaded_file.name[-3:] != "mp4":
        st.error("Please upload a valid .mp4 file")
        return 0

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    
    model = yolo.yolov5()
    annotations_df = predict(tfile.name, model)
    if not os.path.exists('outputs'):
        os.mkdir("outputs")
    postprocess(tfile.name, annotations_df, save_path='outputs')
    st.write("DONE")
        

    
    # fig, ax = plt.subplots()
    # display.display_heatmap(annotations_df, current_frame, fig, ax)
    # ax.set_axis_off()
    # st.pyplot(fig)

    # st.session_state['metric_video_duration'] = 35
    # st.session_state['metric_pedestrian_count'] = 68
    
    # capture.release()