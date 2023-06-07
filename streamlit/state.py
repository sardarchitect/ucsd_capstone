import streamlit as st

def initialize():
    print("Running initalize")
    if 'is_run' not in st.session_state:
        st.session_state['is_run'] = False

    if 'is_plot' not in st.session_state:
        st.session_state['is_plot'] = False

    if 'display_type' not in st.session_state:
        st.session_state["display_type"] = "Video"
    
    if 'analysis_type' not in st.session_state:
        st.session_state["analysis_type"] = "Bounding Boxes"

    if 'have_video_dict' not in st.session_state:
        st.session_state["have_video_dict"] = False

    if 'have_preds' not in st.session_state:
        st.session_state["have_preds"] = False

    if 'have_preprocessed' not in st.session_state:
        st.session_state["have_preprocessed"] = False

def set_run(uploaded_file):
    if uploaded_file is None or uploaded_file.name[-3:] != "mp4":
        # Assert and read video
        st.error("Please upload a valid .mp4 file")
        st.session_state['is_run'] == False
    else:
        st.session_state["have_video_dict"] = False
        st.session_state["have_preds"] = False
        st.session_state["have_preprocessed"] = False
        st.session_state['is_run'] = not(st.session_state['is_run'])