import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from streetstudy.common import postprocess

def display_analysis(video_metadata, frame, current_frame_number, preds, analysis_type="None"):
    """
    Display frame along with annotations provided in standardized format

    Keyword Arguments:
    video_path -- Path to video 
    annotations_df -- Pandas DataFrame of video annotations in standard YOLO format
    frame -- specify frame number to extract bounding boxes from that specific frame

    Return:
    None
    """
    fig, ax = plt.subplots()
    ax.set(xlim=(0, video_metadata['width']), ylim=(video_metadata['height'], 0))    
    ax.set_axis_off()
    ax.set_facecolor('b')
    
    if analysis_type == 'heatmap':
        postprocess.heatmap(preds, current_frame_number, ax)
    if analysis_type == 'bounding_boxes':
        postprocess.bounding_boxes(preds, current_frame_number, ax)
    if analysis_type == 'directional_arrows':
        postprocess.directional_arrows(preds, current_frame_number, ax)
            
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if save_path:
