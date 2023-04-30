import os
import torch 

# resnet18 is the name of entrypoint
def yolov5():
    # Call the model, load pretrained weights
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, 'pedtracker/model/weights/yolov5s.pt')
    
    # torch.hub.set_dir(checkpoint)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model