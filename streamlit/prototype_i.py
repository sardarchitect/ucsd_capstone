import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import cv2 as cv
import matplotlib.pyplot as plt
import torch 
import seaborn as sns
import sys

sys.path.append('/home/sardarchitect/repos/github.com/ucsd_capstone')
from streetstudy.model import yolo
from streetstudy.common import display

def perform_analysis(uploaded_file):
    if uploaded_file is None:
        st.error("Please upload a valid file")
        return 0
    if uploaded_file.name[-3:] != "mp4":
        st.error("Upload a valid .mp4 file")
        return 0
    else:
        st.write('Processing video using YOLOv5')
        model = yolo.yolov5()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        capture = cv.VideoCapture(tfile.name)
        success, current_frame = capture.read()
        annotations_df = yolo.predict_video(model, tfile.name, is_df=True, is_streamlit=True)
        fig, ax = plt.subplots()
        display.display_heatmap(annotations_df, current_frame, fig, ax)
        ax.set_axis_off()
        st.pyplot(fig)

        st.session_state['metric_video_duration'] = 35
        st.session_state['metric_pedestrian_count'] = 68
        
        capture.release()


####################################### APP #######################################
# Upload File
st.title("StreetStudy")
st.caption("Pedestrian Analysis Using YOLOv5")
st.divider()
uploaded_file = st.file_uploader(label='Upload a video to perform analysis')
start_button = st.button('Start')

st.session_state['metric_video_duration'] = None
st.session_state['metric_pedestrian_count'] = None

# If user clicks "Go", start inference and output heatmap along with analysis dashboard

if start_button:
    perform_analysis(uploaded_file)

met_col1, met_col2 = st.columns(2)
met_col1.metric("Video Duration", st.session_state['metric_video_duration'])
met_col2.metric("Pedestrian Count", st.session_state['metric_pedestrian_count'])

dash_row1col1, dash_row1col2 = st.columns(2)
dash_row1col1.subheader('Footpath Density')
dash_row1col1.radio('Filter by', options=["Objects","Events"], key="activity")
dash_row1col2.subheader('Activity Map')
dash_row1col2.selectbox('Filter by', options=["obj_1","obj_2", "obj_3", "obj_4"])

dash_row2col1, dash_row2col2 = st.columns(2)
dash_row2col1.subheader('Direction of Travel')
dash_row2col2.subheader('Dwell Times')

