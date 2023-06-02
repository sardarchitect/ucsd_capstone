import matplotlib.pyplot as plt
import matplotlib.patches
import cv2 as cv
# from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd


def display_raw_video(video_path):
    """
    Displays unannotated (raw) video on default media player

    Keyword Arguments:
    video_path -- path of video to display
    
    Return:
    None
    """
    print("Displaying current video:", video_path)
    capture = cv.VideoCapture(video_path)       
    while True:
        _, current_frame = capture.read()
        # Display current frame number
        current_frame_number = capture.get(cv.CAP_PROP_POS_FRAMES)
        cv.putText(img=current_frame, text=str(current_frame_number), org=(50,50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        cv.imshow('Video', current_frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()

def get_bbox(annotations_df, frame):
    """
    Get bounding boxes for displaying annotations

    Keyword Arguments:
    annotations_df -- Pandas DataFrame of video annotations in standard YOLO format
    frame_number -- specify frame number to extract bounding boxes from that specific frame

    Return:
    bbox_df -- Pandas DataFrame with annotations converted to a more appropriate format for display
    """
    bbox_df = pd.DataFrame(columns=['object_id', 'bbox_lefttop_x', 'bbox_lefttop_y', 'bbox_width', 'bbox_height'])
    video_df = annotations_df[annotations_df['current_frame'] == frame]
    bbox_df['object_id'] = video_df['object_id']
    bbox_df['bbox_lefttop_x'] = video_df['bbox_center_x'] - (video_df['bbox_width'] / 2)
    bbox_df['bbox_lefttop_y'] = video_df['bbox_center_y'] - (video_df['bbox_height'] / 2)
    bbox_df['bbox_width'] = video_df['bbox_width']
    bbox_df['bbox_height'] = video_df['bbox_height']
    return bbox_df

def display_annotated_frame(video_path, annotations_df, frame):
    """
    Display frame along with annotations provided in standardized format

    Keyword Arguments:
    video_path -- Path to video 
    annotations_df -- Pandas DataFrame of video annotations in standard YOLO format
    frame -- specify frame number to extract bounding boxes from that specific frame

    Return:
    None
    """
    capture = cv.VideoCapture(video_path)
    image_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    image_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set(xlim=(0, image_width), ylim=(image_height, 0))
    for _, bbox in get_bbox(annotations_df, frame).iterrows():
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (bbox['bbox_lefttop_x'], bbox['bbox_lefttop_y']), 
                bbox['bbox_width'], 
                bbox['bbox_height'], 
                rotation_point='xy',
                facecolor='none', 
                ec='r', 
                lw=1)
        )
    while True:
        success, current_frame = capture.read()
        current_frame_number = capture.get(cv.CAP_PROP_POS_FRAMES)
        if not success:
            break
        if current_frame_number != frame:
            continue
        ax.imshow(current_frame)
        break
    capture.release()

def display_heatmap(annotations_df, frame, fig=None, ax=None, save_path=None):
    """
    Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

    Keyword Arguments:
    annotations_df -- Annotations in standard YOLO format
    frame -- Specify background frame to display

    Return:
    None
    """
    if not fig:
        fig, ax = plt.subplots()
    feet_x = annotations_df['bbox_center_x']
    feet_y = annotations_df['bbox_center_y'] + (annotations_df['bbox_height'] / 2)

    sns.kdeplot(x=feet_x, y=feet_y, thresh=0, levels=50, alpha=0.2, fill=True, ax=ax, cmap='hot')
    ax.set_axis_off()
    
    fig.set_dpi(300)
    fig.set_figheight(15)
    fig.set_figwidth(35)

    fig.gca().invert_yaxis()
    ax.imshow(frame)
    
    if save_path:
        fig.savefig(save_path)
        plt.close()
        return 1
    return ax