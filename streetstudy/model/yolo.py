import os
import cv2 as cv
import torch 
from tqdm import tqdm

def yolov5(custom=False, conf=0.25):

    if custom:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='../../ultralytics/yolov5/runs/train/streetstudy_yolov52/weights/best.pt')
        model.conf = conf
        model.classes = [0]
        return model
    
    model = torch.hub.load('ultralytics/yolov5', model='yolov5s')
    model.conf = conf
    model.classes = [0]
    return model

def predict_video(model, video_path):
    capture = cv.VideoCapture(video_path)
    annotations = torch.zeros((1, 7), device='cuda')
    total_frame_number = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    i = 0
    for current_frame_number in tqdm(range(total_frame_number)):
        _, frame = capture.read()
        results = model(frame)
        current_frame_number_col = torch.ones((results.xywh[0].shape[0]), device='cuda') * current_frame_number
        list_of_preds = torch.column_stack((current_frame_number_col, results.xywh[0]))
        annotations = torch.row_stack((annotations, list_of_preds))
    capture.release()
    return annotations