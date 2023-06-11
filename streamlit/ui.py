import os
import streamlit as st
import state
from streetstudy.common import postprocess
import streamlit_elements
import compute

def header():
    st.title("🚶🏽‍♀️🚶🚶🏿‍♀️ StreetStudy 🚶🏿‍♂️🚶🏼‍♀️🚶🏽")
    st.subheader("Analyze Pedestrian Traffic Using YOLOv5")
    st.markdown("Thank you for checking out StreetStudy! The goal of this project is to allow urban designers and planners to analyze \
                outdoor pedestrian foot traffic patterns and behaviors in the hopes to help optimize public spaces for pedestrian needs.\
                The app aims to provide desingers granular insights to create more pedestrian-friendly, accessible, and sustainable urban areas")
    st.divider()

def sidebar():
    with st.sidebar:
        st.subheader("Usage")
        st.markdown("1. Upload a .mp4 video (max. size 500mb)")
        st.markdown("2. Click Run")
        st.markdown("3. Filter model results to display various analysis visualizations")
        
        uploaded_file = st.file_uploader(label='Upload a file', label_visibility="hidden")
        st.button(label=('Run' if st.session_state['is_run'] == False else "Stop"), args=[uploaded_file], on_click=state.set_run)
        # CITATIONS 
        st.subheader("Citations")
        return uploaded_file

def analysis_dashboard():
    dash_c1, dash_c2 = st.columns([3, 1])
    with dash_c1:
        if st.session_state["display_type"] == "video":
            st.video(os.path.join(st.session_state["save_path"], st.session_state["analysis_type"], "output.mp4"))
        if st.session_state["display_type"] == "interactive_plot":
            compute.show_plot(50)
        met_c1, met_c2 = st.columns([3,1])
        with met_c1:
            st.subheader("Dwell Times")
            compute.plot_dwell()
        with met_c2:
            st.metric(label="Pedestrian Count", value=12)
            st.metric(label="Pedestrian Count", value=35)
            st.metric(label="Pedestrian Count", value=98)
        
    with dash_c2:
        st.subheader("Analysis Options")
        with st.form("dashboard_form"):
            radio_display_type = st.radio("Display Type", options=["video", "interactive_plot"], key="radio_display_type")
            radio_analysis_type = st.radio("Analysis Type", options=["heatmap", "bounding_boxes", "directional_arrows"], key="radio_analysis_type")
            if st.form_submit_button("Render"):
                st.session_state["display_type"] = radio_display_type
                st.session_state["analysis_type"] = radio_analysis_type