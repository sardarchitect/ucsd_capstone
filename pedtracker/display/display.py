import cv2 as cv
from virat.virat import get_annotations

def get_bbox(df, current_frame):
    bbox_list = []
    # GET BBOX COORDS FOR DISPLAY
    for index, row in (df[df['current_frame'] == current_frame]).iterrows():
        bbox_lefttop_x = row["bbox_lefttop_x"]
        bbox_lefttop_y = row["bbox_lefttop_y"]
        bbox_width = row["bbox_width"]
        bbox_height = row["bbox_height"]
        bbox_rightbottom_x = bbox_lefttop_x + bbox_width
        bbox_rightbottom_y = bbox_lefttop_y + bbox_height
        bbox_list.append([bbox_lefttop_x, bbox_lefttop_y, bbox_rightbottom_x, bbox_rightbottom_y, row["object_type"]])
    return bbox_list

def display_raw_video(video_path):
    # Displays videos provided the path to the video along with the current frame
    print("Displaying current video:", video_path)
    capture = cv.VideoCapture(video_path)       
    while True:
        isTrue, current_frame = capture.read()
        # Display current frame number
        current_frame_number = capture.get(cv.CAP_PROP_POS_FRAMES)
        cv.putText(img=current_frame, text=str(current_frame_number), org=(50,50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        cv.imshow('Video', current_frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()

def display_annotated_video(video_path, virat=True):
    # Displays annoted videos provided the path to the video and to the annotations along with the current frame.
    # If video from VIRAT, then finds annotation for the specified video using in-built functions
    if virat:
        annotations_df = get_annotations(video_path, virat)
    print("Displaying current video:", video_path)
    capture = cv.VideoCapture(video_path)       
    while True:
        isTrue, current_frame = capture.read()
        # Display current frame number
        current_frame_number = capture.get(cv.CAP_PROP_POS_FRAMES)
        cv.putText(img=current_frame, text=str(current_frame_number), org=(50,50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        # GET BBOX LIST FOR CURRENT FRAME
        bbox_list = get_bbox(annotations_df, current_frame_number)
        # DRAW RECTANGLE FOR EVERY BBOX IN FRAME
        current_frame_overlay = current_frame.copy() # Duplicate to apply transparency mask
        for bbox in bbox_list:
            bbox_color = (255,0,0) if bbox[4] == 1 else (255,255,255)
            cv.rectangle(img=current_frame_overlay, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=bbox_color, thickness=-1)
            cv.rectangle(img=current_frame, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=bbox_color, thickness=1)
            alpha = 0.05 # Transparency factor
            current_frame = cv.addWeighted(current_frame_overlay, alpha, current_frame, 1 - alpha, 0)
            cv.putText(img=current_frame, text=str(bbox[4]), org=(bbox[0], bbox[1]), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
        # DRAW CURRENT FRAME NUMBER
        cv.putText(img=current_frame, text=str(current_frame_number), org=(50,50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        cv.imshow('Video', current_frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()