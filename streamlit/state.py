from streetstudy.common import utils 
import streamlit as st
import shutil

def initialize():
    print("Running initalize")
    if "save_path" not in st.session_state:
        st.session_state["save_path"] = '.data_cache/'
        shutil.rmtree(st.session_state["save_path"])
        utils.make_dir(st.session_state["save_path"])

    if 'is_run' not in st.session_state:
        st.session_state['is_run'] = False

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

def set_run(uploaded_file):
    if uploaded_file is None or uploaded_file.name[-3:] != "mp4":
        # Assert and read video
        st.error("Please upload a valid .mp4 file")
        st.session_state['is_run'] == False
    else:
        st.session_state["cache_video_dict"] = False
        st.session_state["cache_preds"] = False
        st.session_state["cache_postprocess"] = False
        st.session_state['is_run'] = not(st.session_state['is_run'])