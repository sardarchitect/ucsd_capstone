import cv2 as cv
import os
import torch 

def yolov5():
    # Call the model, load pretrained weights
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, 'pedtracker/model/checkpoints/yolov5s.pt')
    
    # torch.hub.set_dir(checkpoint)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/sardarchitect/repos/github.com/ucsd_capstone/ultralytics/yolov5/runs/train/streetstudy_yolov52/weights/best.pt')
    model.conf = 0.25
    model.classes = [0]
    return model