import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import cv2 as cv
# from scipy.stats import gaussian_kde
import seaborn as sns
import streamlit as st
import pandas as pd

def get_bbox(video_annotations, frame_number):
    bbox_df = pd.DataFrame(columns=['object_id', 'bbox_lefttop_x', 'bbox_lefttop_y', 'bbox_width', 'bbox_height'])
    video_df = video_annotations[video_annotations['current_frame'] == frame_number]
    bbox_df['object_id'] = video_df['object_id']
    bbox_df['bbox_lefttop_x'] = video_df['bbox_center_x'] - (video_df['bbox_width'] / 2)
    bbox_df['bbox_lefttop_y'] = video_df['bbox_center_y'] - (video_df['bbox_height'] / 2)
    bbox_df['bbox_width'] = video_df['bbox_width']
    bbox_df['bbox_height'] = video_df['bbox_height']
    return bbox_df

def display_raw_video(video_path):
    # Displays video from path
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

def display_annotated_video(video_path, annotations_df, num_frames=1000):
    # Displays video from path along with provided annotations
    print("Displaying current video:", video_path)
    capture = cv.VideoCapture(video_path)       
    while True:
        _, current_frame = capture.read()
        # Display current frame number
        current_frame_number = capture.get(cv.CAP_PROP_POS_FRAMES)
        if current_frame_number >= num_frames:
            break
        
        cv.putText(img=current_frame, text=str(current_frame_number), org=(50,50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        # Get bbox_list at current_frame
        bbox_list = get_bbox(annotations_df, current_frame_number)
        
        # Draw bbox
        current_frame_overlay = current_frame.copy() # Duplicate to apply transparency mask
        for bbox in bbox_list:
            bbox_color = (255,0,0) if bbox[4] == 1 else (255,255,255)
            cv.rectangle(img=current_frame_overlay, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=bbox_color, thickness=-1)
            cv.rectangle(img=current_frame, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=bbox_color, thickness=1)
            alpha = 0.05 # Transparency factor
            current_frame = cv.addWeighted(current_frame_overlay, alpha, current_frame, 1 - alpha, 0)
            cv.putText(img=current_frame, text=str(bbox[4]), org=(bbox[0], bbox[1]), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
        
        # Draw current_frame_number
        cv.putText(img=current_frame, text=str(current_frame_number), org=(50,50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        
        # Show video
        cv.imshow('Video', current_frame)
        key = cv.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv.waitKey(0)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv.destroyAllWindows()

def display_annotated_frame(video_path, video_annotations, frame_number):
    capture = cv.VideoCapture(video_path)
    
    image_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    image_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set(xlim=(0, image_width), ylim=(image_height, 0))
    for _, bbox in get_bbox(video_annotations, frame_number).iterrows():
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
        if current_frame_number != frame_number:
            continue
        ax.imshow(current_frame)
        break
    capture.release()

def display_heatmap(frame, df):
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x='x',y='y', thresh=0, levels=50, ax=ax, fill=True, alpha=0.2, cmap='hot')
        fig.gca().invert_yaxis()
        plt.imshow(frame)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        