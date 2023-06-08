import os
import tempfile

import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st

import postprocess
from streetstudy.model import yolo
from streetstudy.common import utils
import numpy as np
from sort.sort import Sort


def video_metadata(uploaded_file):
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
    st.session_state["have_video_dict"] = True
    return video_dict

def predict(video_dict):    
    loading_bar_text = "Inference in progress. This may take a couple minutes. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    # Init model
    model = yolo.yolov5()

    #Store annotations
    preds_list = np.empty((0, 6))
    # Init tracker
    sort = Sort(max_age=5, min_hits=3, iou_threshold=0.2)
    # Init video loop
    capture = cv.VideoCapture(video_dict['path'])
    while True:
        current_frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        success, frame = capture.read()
        if not success:
            break
    
        # Predict
        predictions = model(frame).xyxy[0]
        tracked_predictions = sort.update(predictions[:, :4].cpu().numpy())
        current_frame_number_np = np.ones((tracked_predictions.shape[0], 1)) * current_frame_number
        tracked_predictions = np.hstack((current_frame_number_np, tracked_predictions))
        preds_list = np.vstack((preds_list, tracked_predictions))    
        loading_bar.progress(current_frame_number/video_dict["length"], text=loading_bar_text)
    loading_bar.empty()
    capture.release()
    st.session_state["have_preds"] = True
    return preds_list

def postprocess_videos(video_dict, preds, SKIP=50):
    postprocess_list = ["heatmap", "bounding_boxes", "directional_arrows"]
    for path in postprocess_list:
        utils.make_dir(os.path.join(st.session_state["save_path"], path))

    loading_bar_text = "Postprocessing in progress. This may take a couple minutes. Please wait."
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
            fig.savefig(os.path.join(st.session_state["save_path"], item, video_dict['path'][-6:] + "_" + str(current_frame_number)))
            plt.close()
            
        loading_bar.progress(current_frame_number/video_dict["length"], text=loading_bar_text)
    
    for folder in os.listdir(st.session_state["save_path"]):
        utils.generate_video(os.path.join(st.session_state["save_path"], folder))
    
    loading_bar.empty()
    capture.release()
    st.session_state["have_preprocessed"] = True
    st.session_state["is_plot"] = True
    
    
