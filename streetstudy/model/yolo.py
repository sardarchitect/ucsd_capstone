import os
import cv2 as cv
import torch 
from tqdm import tqdm
import pandas as pd

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
    model = torch.hub.load('ultralytics/yolov5', model='yolov5s')
    model.conf = conf
    model.classes = [0] # Detect "Person" class only
    return model

def predict_video(model, video_path, is_df=False):
    """
    Get annotations from a video using specified model

    model -- YOLOv5 model for inference
    video_path -- path to video
    is_df -- flag to return as Pandas DataFrame if True (default False)

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