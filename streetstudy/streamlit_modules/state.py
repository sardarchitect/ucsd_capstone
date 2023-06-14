# Imports
import streamlit as st
import shutil
import os

# Import app modules
from streetstudy.common import utils 

def initialize():
    """
    Initialize the session state variables.
    """
    if "save_path" not in st.session_state:
        st.session_state["save_path"] = '.data_cache/'
        if os.path.exists(st.session_state["save_path"]):
            shutil.rmtree(st.session_state["save_path"])
        utils.make_dir(st.session_state["save_path"])

    if 'running' not in st.session_state:
        st.session_state['running'] = False

    if 'is_plot' not in st.session_state:
        st.session_state['is_plot'] = False

    if 'display_type' not in st.session_state:
        st.session_state["display_type"] = "video"
    
    if 'analysis_type' not in st.session_state:
        st.session_state["analysis_type"] = "bounding_boxes"

    if 'cache_video_dict' not in st.session_state:
        st.session_state["cache_video_dict"] = False

    if 'cache_preds' not in st.session_state:
        st.session_state["cache_preds"] = False

    if 'cache_postprocess' not in st.session_state:
        st.session_state["cache_postprocess"] = False

    if 'unique_objs' not in st.session_state:
        st.session_state["unique_objs"] = 0

    if 'total_dwell_time' not in st.session_state:
        st.session_state["total_dwell_time"] = 0

    if 'video_length' not in st.session_state:
        st.session_state["video_length"] = 0

    if 'current_frame_number' not in st.session_state:
        st.session_state["current_frame_number"] = 0
    

def running(uploaded_file):
    """
    Start or stop the "running" flag based on current state and whether uploaded file is ".mp4".

    Args:
        uploaded_file (Object): Uploaded video file object.
    """
    if uploaded_file is None or uploaded_file.name[-3:] != "mp4":
        # Assert and read video
        st.error("Please upload a valid .mp4 file")
        st.session_state['running'] == False
    else:
        st.session_state["cache_video_dict"] = False
        st.session_state["cache_preds"] = False
        st.session_state["cache_postprocess"] = False
        st.session_state['running'] = not(st.session_state['running'])