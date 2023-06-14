# Imports
import os
import tempfile
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st
import plotly.express as plty_exp
import numpy as np

# Import app modules
from streetstudy.model import yolo
from streetstudy.common import utils
from streetstudy.common import postprocess

def get_video_metadata(uploaded_file):
    """
    Get metadata for the uploaded video file.

    Args:
        uploaded_file (Object): Uploaded video file object.
    
    Returns:
        video_metadata (dict): Video metadata.
    """
    video_metadata_path = os.path.join(st.session_state['save_path'], 'video_metadata.pkl')

    if not os.path.exists(video_metadata_path):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_metadata = utils.get_video_metadata(tfile.name)
        with open(video_metadata_path, 'wb') as f:
            pickle.dump(video_metadata, f)
    else:
        with open(video_metadata_path, 'rb') as f:
            video_metadata = pickle.load(f)

    st.session_state['cache_video_metadata'] = True
    st.session_state['video_length'] = video_metadata['length']

    return video_metadata

def predict(video_metadata):
    """
    Run predictions on video frames.

    Keyword args:
        video_metadata (dict): Video metadata.

    Returns:
        model.predictions (np.ndarray): Predicted results.
    """
    preds_path = os.path.join(st.session_state['save_path'], 'preds.pkl')
    if os.path.exists(preds_path):
        preds = np.load(preds_path, allow_pickle=True)
        return preds 
    
    loading_bar_text = 'Inference in progress. This may take a couple minutes. Please wait.'
    loading_bar = st.progress(0, text=loading_bar_text)
    
    # Initialize model
    model = yolo.YoloModel(conf=0.25)
    
    frame_iterator = iter(utils.frame_generator(path=video_metadata['path']))
    for frame_number in range(video_metadata['length']):
        frame = next(frame_iterator)
        model.predict(frame, frame_number)
        loading_bar.progress(frame_number / video_metadata['length'], text=loading_bar_text)

    loading_bar.empty()

    st.session_state['cache_preds'] = True

    with open(preds_path, 'wb') as f:
        pickle.dump(model.predictions, f)

    return model.predictions

def postprocess_videos(video_metadata, preds, SKIP=25):
    """
    Perform postprocessing on the video frames using model predictions.
    
    Keyword args:
        video_metadata (dict): Video metadata.
        preds (np.ndarray): Predicted results.
        SKIP (int, optional): Frame skipping interval. Defaults to 25.
    """
    st.session_state['unique_objs'] = len(np.unique(preds[:,5]))
    postprocess_list = ['heatmap', 'bounding_boxes', 'directional_arrows']
    
    for path in postprocess_list:
        utils.make_dir(os.path.join(st.session_state['save_path'], path))

    loading_bar_text = 'Postprocessing in progress. This may take a couple minutes. Please wait.'
    loading_bar = st.progress(0, text=loading_bar_text)
    
    valid_frames = [x for x in range(0, video_metadata['length'], SKIP)]
    capture = cv.VideoCapture(video_metadata['path'])
    count = 0
    
    while True:
        current_frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        success, frame = capture.read()
    
        if not success:
            break
        if current_frame_number not in valid_frames:
            continue
        for item in postprocess_list:
            fig, ax = plt.subplots()
            ax.set(xlim=(0, video_metadata['width']), ylim=(video_metadata['height'], 0))    
            ax.set_axis_off()
            ax.set_facecolor('b')

            if item == 'heatmap':
                postprocess.heatmap(preds, current_frame_number, ax)
            if item == 'bounding_boxes':
                postprocess.bounding_boxes(preds, current_frame_number, ax)
            if item == 'directional_arrows':
                postprocess.directional_arrows(preds, current_frame_number, ax)
            
            ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            fig.savefig(os.path.join(st.session_state['save_path'], item, str(count)), transparent=True)
            plt.close()
            
        loading_bar.progress(current_frame_number/video_metadata['length'], text=loading_bar_text)
        count+=1
    
    for folder in os.listdir(st.session_state['save_path']):
        utils.generate_video(os.path.join(st.session_state['save_path'], folder))
    
    loading_bar.empty()
    capture.release()
    
    st.session_state['cache_postprocess'] = True
    st.session_state['is_plot'] = True
    
def show_interactive_plot(video_metadata, preds):
    """
    Show an interactive plot of a video frame with postprocessed analysis overlay.
    
    Args:
        video_metadata (dict): Video metadata.
        preds (np.ndarray): Predicted results.
    """
    fig, ax = plt.subplots()
    ax.set(xlim=(0, video_metadata['width']), ylim=(video_metadata['height']))
    ax.set_axis_off()
    ax.set_facecolor('b')

    current_frame_number = st.session_state['current_frame_number']

    frame_iterator = iter(utils.frame_generator(path=video_metadata['path']))
    for frame_number in range(video_metadata['length']):
        frame = next(frame_iterator)
        if frame_number == current_frame_number:
            break

        continue

    if st.session_state['analysis_type'] == 'heatmap':
        postprocess.heatmap(ax, preds, current_frame_number)
    if st.session_state['analysis_type'] == 'bounding_boxes':
        postprocess.bounding_boxes(ax, preds, current_frame_number)
    if st.session_state['analysis_type'] == 'directional_arrows':
        postprocess.directional_arrows(ax, preds, current_frame_number)
    
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    st.pyplot(fig)

def plot_dwell():
    """
    Plot a dwelling chart using dummy data.
    """    
    x = np.arange(start=0, stop=100, step=0.5)
    y = np.sin(x)
    fig = plty_exp.line(x=x,y=y, height=300)
    st.plotly_chart(fig, use_container_width=True)