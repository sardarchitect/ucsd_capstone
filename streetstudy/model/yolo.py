import torch 
from streetstudy.common.sort import Sort
import numpy as np

class YoloModel():
    def __init__(self, conf=0.25):
        """
        Get specified YOLO model

        Keyword arguments:
        model -- name of model (default "yolov5s"). Use "custom" if using custom model
        conf -- minimum confidence threshold (default 0.25)
        path -- path to custom model (defaul None)

        Return:
        model -- YOLOv5 model object
        """
        self.model = torch.hub.load('ultralytics/yolov5', model='yolov5s', pretrained=True)
        self.model.conf = conf
        self.model.classes = [0] # Detect "Person" class only
        self.predictions = np.empty((0, 6))
        self.tracker= Sort(max_age=5, min_hits=3, iou_threshold=0.1)      
        
    def predict(self, image, frame_number):
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
        preds = self.model(image).xyxy[0]
        preds = self.tracker.update(preds[:, :4].cpu().numpy())
        temp = np.ones((preds.shape[0], 1)) * frame_number
        preds = np.hstack((temp, preds))
        self.predictions = np.vstack((self.predictions, preds))