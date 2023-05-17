import os
import cv2 as cv
import torch 
from tqdm import tqdm

def yolov5():
    # Call the model, load pretrained weights
    path_custom_weights = '/home/sardarchitect/repos/github.com/ucsd_capstone/ultralytics/yolov5/runs/train/streetstudy_yolov52/weights/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_custom_weights)
    model.conf = 0.25
    model.classes = [0]
    return model

def predict(video_path, display_video=False):
    capture = cv.VideoCapture(video_path)
    model = yolov5()
    annotations = []

    total_frame_number = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frame_number)):
        _, current_frame = capture.read()
        results = model(current_frame)
        annotations.append(results.xyxy.cpu().numpy())

    # DESTROY WINDOWS
    capture.release()
    cv.destroyAllWindows()

    return annotations