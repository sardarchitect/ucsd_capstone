import cv2 as cv
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

def inference(video_path, model_type='yolov5'):
    print("Displaying current video:", video_path)
    capture = cv.VideoCapture(video_path)
    if model_type == 'yolov5':
        model = yolov5()
    
    while True:
        _, current_frame = capture.read()
        results = model(current_frame)
        results.show()
        cv.imshow('Video', current_frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    # DESTROY WINDOWS
    capture.release()
    cv.destroyAllWindows()

    return results