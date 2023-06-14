import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import postprocess

def display_annotated_frame(video_path, video_metadata, frame, preds, current_frame_number, analysis_type, save_path):
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
        postprocess.heatmap(ax, preds, current_frame_number)
    if analysis_type == 'bounding_boxes':
        postprocess.bounding_boxes(ax, preds, current_frame_number)
    if analysis_type == 'directional_arrows':
        postprocess.directional_arrows(ax, preds, current_frame_number)
    
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    fig.savefig(os.path.join(st.session_state['save_path'], item, str(count)), transparent=True)
    plt.close()