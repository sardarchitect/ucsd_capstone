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

import cv2 as cv
from model import yolo
from tqdm import tqdm

def inference(video_path, model_type='yolov5', display_video=False):
    print("Displaying current video:", video_path)
    capture = cv.VideoCapture(video_path)
    if model_type == 'yolov5':
        model = yolo.yolov5()

    # TODO
    # GET ANNOTATIONS FROM MODEL
    # CONVERT INTO STANDARDIZED VIRAT ANNOTATIONS
    # CALL DISPLAY

    annotations = []

    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(total_frames)):
        _, current_frame = capture.read()
        results = model(current_frame)
        annotations.append(results.xyxy.cpu().numpy())

        if display_video == True:
            results.show()
            cv.imshow('Video', current_frame)
            key = cv.waitKey(0)
            while key not in [ord('q'), ord('k')]:
                key = cv.waitKey(0)

            if cv.waitKey(20) & 0xFF == ord('q'):
                break
        
    # DESTROY WINDOWS
    capture.release()
    cv.destroyAllWindows()

    return annotations