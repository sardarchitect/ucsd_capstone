import os
import cv2 as cv
import torch 
from tqdm import tqdm
import pandas as pd
from sort.sort import Sort
import streamlit as st
import numpy as np

def yolov5(model='yolov5s', conf=0.25, path=None):
    """
    Get specified YOLO model

    Keyword arguments:
    model -- name of model (default "yolov5s"). Use "custom" if using custom model
    conf -- minimum confidence threshold (default 0.25)
    path -- path to custom model (defaul None)

    Return:
    model -- YOLOv5 model object
    """
    model = torch.hub.load('ultralytics/yolov5', model='yolov5s', pretrained=True)
    model.conf = conf
    model.classes = [0] # Detect "Person" class only
    return model

def predict_video(model, video_path, is_df=False):
    """
    Get annotations from a video using specified model

    model -- YOLOv5 model for inference
    video_path -- path to video
    is_df -- flag to return as Pandas DataFrame if True (default False)
    is_streamlit -- flag to return loading bar progress if True (default False)

    Return: Model predictions as PyTorch tensors (or Pandas DataFrame if is_df==True)
    Column names:
        0: current_frame
        1: bbox_center_x
        2: bbox_center_y
        3: width
        4: height
        5: conf
        6: class (0 is person class)
    """
    
    capture = cv.VideoCapture(video_path)
    annotations = torch.empty((0, 7), device='cuda')
    total_frame_number = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    for current_frame_number in tqdm(range(total_frame_number)):
        _, frame = capture.read()
        results = model(frame)
        current_frame_number_col = torch.ones((results.xywh[0].shape[0]), device='cuda') * current_frame_number
        list_of_preds = torch.column_stack((current_frame_number_col, results.xywh[0]))
        annotations = torch.row_stack((annotations, list_of_preds))

    capture.release()
    if is_df:
        annotations_df = pd.DataFrame(data=annotations.cpu().numpy(), columns=['current_frame', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height', 'conf', 'class'])
        return annotations_df
    return annotations

def st_predict(video_dict):
    loading_bar_text = "Inference in progress. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    # Init model
    model = yolov5()

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
    return preds_list