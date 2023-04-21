# Imports
import cv2 as cv
from utils import *

# STORE DATA DIRECTORY
data_dir = './data/VIRAT Ground Dataset/'
annotations_dir = data_dir + 'annotations/'
videos_dir = data_dir + 'videos_original/'

# SELECT CURRENT VIDEO
current_video = "VIRAT_S_000102"

# CREATE DATAFRAMES FROM ANNOTATION TEXT FILES FOR CURRENT VIDEO
video_objects_df = virat_txt_to_df(annotations_dir, current_video, table_category='objects')

# CREATE VIDEO OBJECT FOR CURRENT VIDEO
capture = cv.VideoCapture(videos_dir+current_video+".mp4")

# READ AND DISPLAY FRAMES AND ASSOCIATED ANNOTATIONS
while True:
    isTrue, current_frame = capture.read()

    # GET CURRENT FRAME
    current_frame_number = capture.get(cv.CAP_PROP_POS_FRAMES)

    # GET BBOX LIST FOR CURRENT FRAME
    bbox_list = get_bbox(video_objects_df, current_frame_number)

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

    # DISPLAY VIDEO
    cv.imshow('Video', current_frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# DESTROY WINDOWS
capture.release()
cv.destroyAllWindows()