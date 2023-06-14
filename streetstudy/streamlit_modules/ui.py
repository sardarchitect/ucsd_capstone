# Imports
import os

# Import app modules
import streamlit as st
import streetstudy.streamlit_modules.state as state
import streetstudy.streamlit_modules.compute as compute

def header():
    """
    Display the header section of the app.
    """
    st.title("🚶🏽‍♀️🚶🚶🏿‍♀️ StreetStudy 🚶🏿‍♂️🚶🏼‍♀️🚶🏽")
    st.subheader("Analyze Pedestrian Traffic Using YOLOv5")
    st.markdown("Thank you for checking out StreetStudy! The goal of this project is to allow urban designers and planners to analyze \
                outdoor pedestrian foot traffic patterns and behaviors in the hopes to help optimize public spaces for pedestrian needs.\
                The app aims to provide designers granular insights to create more pedestrian-friendly, accessible, and sustainable urban areas.")
    st.divider()

def sidebar():
    """
    Display the sidebar section of the app and handle file uploading.
    
    Returns:
        uploaded_file (Object): Uploaded video file object.
    """
    with st.sidebar:
        st.subheader("Usage")
        st.markdown("1. Upload a .mp4 video (max. size 500mb)")
        st.markdown("2. Click Run")
        st.markdown("3. Filter model results to display various analysis visualizations")
        
        uploaded_file = st.file_uploader(label='Upload a file', label_visibility="hidden", type="mp4")
        st.button(label=('Run' if st.session_state['running'] == False else "Stop"), args=[uploaded_file], on_click=state.running)
        st.button(label='Demo', args=['demo'], on_click=state.running)
        
        st.subheader("Citations")

        return uploaded_file

def analysis_dashboard(video_metadata, preds):
    """
    Display the analysis dashboard section of the app.
    
    Args:
        video_metadata (dict): Dictionary containing video metadata.
        preds (np.ndarray): Numpy array containing model predictions.
    """
    dash_c1, dash_c2 = st.columns([3, 1])
    with dash_c2:
        st.subheader("Analysis Options")
        with st.form("dashboard_form"):
            radio_display_type = st.radio("Display Type", options=["video", "interactive_plot"], key="radio_display_type")
            radio_analysis_type = st.radio("Analysis Type", options=["heatmap", "bounding_boxes", "directional_arrows"], key="radio_analysis_type", index=1)
            slider_current_frame = st.slider("Current Frame Number", min_value=0, max_value=int(max(preds[:,0])), step=1)
            
            submitted = st.form_submit_button("Render")

            if submitted:
                st.session_state["display_type"] = radio_display_type
                st.session_state["analysis_type"] = radio_analysis_type
                st.session_state["current_frame_number"] = slider_current_frame
                
    with dash_c1:
        if st.session_state["display_type"] == "video":
            st.video(os.path.join(st.session_state["save_path"], st.session_state["analysis_type"], "output.mp4"))
        if st.session_state["display_type"] == "interactive_plot":
            compute.show_interactive_plot(video_metadata, preds)
        
        met_c1, met_c2 = st.columns([3,1])
        with met_c1:
            st.subheader("Dwell Times")
            compute.plot_dwell()
        
        with met_c2:
            st.metric(label="Video Length", value=st.session_state['video_length'])
            st.metric(label="Pedestrian Count", value=st.session_state['unique_objs'])
            st.metric(label="Total Dwell Time", value=st.session_state['total_dwell_time'])