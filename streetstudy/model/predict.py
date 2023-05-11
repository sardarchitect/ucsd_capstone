import cv2 as cv
from streetstudy.model import yolov5

def inference(video_path, model_type='yolov5'):
    print("Displaying current video:", video_path)
    capture = cv.VideoCapture(video_path)
    if model_type == 'yolov5':
        model = yolov5()

    # TODO
    # GET ANNOTATIONS FROM MODEL
    # CONVERT INTO STANDARDIZED VIRAT ANNOTATIONS
    # CALL DISPLAY
        
    while True:
        _, current_frame = capture.read()
        results = model(current_frame)
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

    return results