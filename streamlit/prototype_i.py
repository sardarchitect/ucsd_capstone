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


# Latest model
model = yolo.yolov5()

# Upload File
uploaded_file = st.file_uploader(label='Upload a video')

# If user clicks "Go", start inference and output heatmap along with analysis dashboard
if st.button('Go'):
    if uploaded_file is not None:
        st.write('Processing video using Yolo v5s')
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
    
        capture = cv.VideoCapture(tfile.name)
        image_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        image_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

        annotations = torch.zeros((1, 7), device='cuda')
        total_frame_number = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        loading_bar_text = "Inference in progress. Please wait."
        loading_bar = st.progress(0, text=loading_bar_text)

        for current_frame_number in range(total_frame_number):
            _, frame = capture.read()
            results = model(frame)
            current_frame_number_col = torch.ones((results.xywh[0].shape[0]), device='cuda') * current_frame_number
            list_of_preds = torch.column_stack((current_frame_number_col, results.xywh[0]))
            annotations = torch.row_stack((annotations, list_of_preds))
            loading_bar.progress(current_frame_number/total_frame_number, text=loading_bar_text)
            # break

        capture.release()

        x = (annotations[:, 1] + (annotations[:, 3] / 2)).cpu().numpy()
        y = (annotations[:, 2] + (annotations[:, 4])).cpu().numpy()
        df = np.column_stack((x, y))

        df = pd.DataFrame(df, columns=['x', 'y'])
        
        fig, ax = plt.subplots()

        sns.kdeplot(data=df, x='x',y='y', thresh=0, levels=50, ax=ax, fill=True, alpha=0.2, cmap='hot')
        fig.gca().invert_yaxis()
        plt.imshow(frame)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        st.pyplot(fig)