# Imports
import torch 
import numpy as np

# Import modules
from streetstudy.common.sort import Sort

class YoloModel():
    def __init__(self, conf=0.25):
        """
        Initialize YOLOv5 model with specified `conf` (confidence threshold)

        Keyword arguments:
            conf (float): Minimum confidence threshold (default 0.25)
        """
        self.model = torch.hub.load('ultralytics/yolov5', model='yolov5s', pretrained=True)
        self.model.conf = conf
        self.model.classes = [0] # Detect "Person" class only
        self.predictions = np.empty((0, 6))
        self.tracker= Sort(max_age=5, min_hits=3, iou_threshold=0.1)      
        
    def predict(self, image, frame_number):
        """
        Get annotations from the YOLOv5 model for a given image.

        Args:
            image: Input image for object detection
            frame_number: Frame number of the image

        Returns:
            None

        Note: self.predictions is a np.ndarray where the columns are formatted in the following manner:
            0: current_frame
            1: bbox_topleft_x
            2: bbox_topleft_y
            3: bbox_bottomright_x
            4: bbox_bottomright_y
            5: conf
            6: class (0 is person class)
        """
        preds = self.model(image).xyxy[0]
        preds = self.tracker.update(preds[:, :4].cpu().numpy())
        frame_number_column = np.ones((preds.shape[0], 1)) * frame_number
        preds = np.hstack((frame_number_column, preds))
        self.predictions = np.vstack((self.predictions, preds))