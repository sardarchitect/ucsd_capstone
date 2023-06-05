import os
import tempfile
import cv2 as cv
import matplotlib.pyplot as plt
import streamlit as st

from streetstudy.model import yolo
from streetstudy.common import utils
from streetstudy.common import display

def plot_interactive(option, video_dict, preds):
    fig, ax = plt.subplots()
    ax.set(xlim=(0, video_dict['width']), ylim=(video_dict['height'], 0))
    if option == "Heatmap":
        display.st_heatmap(ax, preds, video_dict['length'])
    if option == "Directional Flow":
        display.st_directional_arrows(ax, preds, video_dict['length'])
    # ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    st.pyplot(fig)
    plt.close()

def postprocess(pp_type, video_dict, current_frame_number, preds, frame, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    fig, ax = plt.subplots()
    ax.set(xlim=(0, video_dict['width']), ylim=(video_dict['height'], 0))
    
    if pp_type == "heatmap":
        display.st_heatmap(ax, preds, current_frame_number)
    if pp_type == "bbox":
        display.st_display_bbox(ax, preds, current_frame_number)
    if pp_type == "arrows":
        display.st_directional_arrows(ax, preds, current_frame_number)
    
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    fig.savefig(os.path.join(save_path, str(current_frame_number)))
    plt.close()
    return

def pipeline(uploaded_file, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # Assert and read video
    if uploaded_file is None or uploaded_file.name[-3:] != "mp4":
        st.error("Please upload a valid .mp4 file")
        return 0
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Video metadata capture
    capture = cv.VideoCapture(tfile.name)
    video_dict = {}
    video_dict["path"] = tfile.name
    video_dict["fps"] = capture.get(cv.CAP_PROP_FPS)
    video_dict["length"] = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    video_dict["height"] = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_dict["width"] = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    capture.release()

    preds = yolo.st_predict(video_dict)
    
    # Postprocess
    SKIP = 50
    loading_bar_text = "Postprocessing in progress. Please wait."
    loading_bar = st.progress(0, text=loading_bar_text)
    capture = cv.VideoCapture(video_dict['path'])
    valid_frames = [x for x in range(0, video_dict['length'], SKIP)]
    
    while True:
        current_frame_number = int(capture.get(cv.CAP_PROP_POS_FRAMES))
        success, frame = capture.read()
        if not success:
            break
        if current_frame_number not in valid_frames:
            continue
        postprocess("heatmap", video_dict, current_frame_number, preds, frame, save_path=os.path.join(save_path, "heatmap"))
        postprocess("bbox", video_dict, current_frame_number, preds, frame, save_path=os.path.join(save_path, "bbox"))
        postprocess("arrows", video_dict, current_frame_number, preds, frame, save_path=os.path.join(save_path, "arrows"))
        loading_bar.progress(current_frame_number/video_dict["length"], text=loading_bar_text)
    
    loading_bar.empty()
    capture.release()
    
    for folder in os.listdir(save_path):
        utils.generate_video(os.path.join(save_path, folder))
    
    return video_dict, preds