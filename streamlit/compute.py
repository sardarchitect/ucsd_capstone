import os
import tempfile

import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st
import plotly as plty
import plotly.express as plty_exp
import numpy as np


from streetstudy.model import yolo
from streetstudy.common import utils
from streetstudy.common import postprocess
import numpy as np
from sort.sort import Sort

def get_video_metadata(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_dict = utils.get_video_metadata(tfile.name)
    st.session_state["cache_video_dict"] = True
    return video_dict

def predict(video_dict):
    loading_bar_text = "Inference in progress. This may take a couple minutes. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    # Init model
    model = yolo.YoloModel(conf=0.25)
    
    frame_iterator = iter(utils.frame_generator(path=video_dict['path']))
    
    for frame_number in range(video_dict['length']):
        frame = next(frame_iterator)
        model.predict(frame, frame_number)
        loading_bar.progress(frame_number/video_dict["length"], text=loading_bar_text)

    loading_bar.empty()

    st.session_state["cache_preds"] = True
    return model.predictions

def postprocess_videos(video_dict, preds, SKIP=50):
    postprocess_list = ["heatmap", "bounding_boxes", "directional_arrows"]
    for path in postprocess_list:
        utils.make_dir(os.path.join(st.session_state["save_path"], path))

    loading_bar_text = "Postprocessing in progress. This may take a couple minutes. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    valid_frames = [x for x in range(0, video_dict['length'], SKIP)]
    
    capture = cv.VideoCapture(video_dict['path'])
    while True:
        current_frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        success, frame = capture.read()
        if not success:
            break
        if current_frame_number not in valid_frames:
            continue

        for item in postprocess_list:
            fig, ax = plt.subplots()
            ax.set(xlim=(0, video_dict['width']), ylim=(video_dict['height'], 0))    
            ax.set_axis_off()
            ax.set_facecolor("b")

            if item == "heatmap":
                postprocess.heatmap(ax, preds, current_frame_number)
            if item == "bounding_boxes":
                postprocess.bounding_boxes(ax, preds, current_frame_number)
            if item == "directional_arrows":
                postprocess.directional_arrows(ax, preds, current_frame_number)
            
            ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB), alpha=0.9)
            fig.savefig(os.path.join(st.session_state["save_path"], item, video_dict['path'][-6:] + "_" + str(current_frame_number)), transparent=True)
            plt.close()
            
        loading_bar.progress(current_frame_number/video_dict["length"], text=loading_bar_text)
    
    for folder in os.listdir(st.session_state["save_path"]):
        utils.generate_video(os.path.join(st.session_state["save_path"], folder))
    
    loading_bar.empty()
    capture.release()
    st.session_state["cache_postprocess"] = True
    st.session_state["is_plot"] = True
    
def show_plot(pix):
    img_rgb = np.random.randint(low=0, high=255, size=(pix, pix))
    fig = plty_exp.imshow(img_rgb)
    fig.add_annotation(text=st.session_state["display_type"], showarrow=False, x=-1, y=-1)
    st.plotly_chart(fig)

def plot_dwell():
    x = np.arange(start=0, stop=100, step=0.5)
    y = np.sin(x)
    fig = plty_exp.line(x=x,y=y, height=300)
    st.plotly_chart(fig, use_container_width=True)