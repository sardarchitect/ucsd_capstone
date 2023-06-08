import streamlit as st

# App Library
import compute
import state
import ui

# CONFIG
st.set_page_config(
    page_title="StreetStudy",
    page_icon="🚶🏽‍♀️🚶",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/sardarchitect/ucsd_capstone',
        'Report a bug': "https://www.arvinder.me",
        'About': "# Thank you! StreetStudy is an *extremely* cool app!"
    }
)
state.initialize()

# HEADER
ui.header()
# SIDEBAR
uploaded_file = ui.sidebar()
# DASHBOARD
if st.session_state['is_run'] == True:
    #TODO: save preds to cache for faster loading
    if st.session_state['have_video_dict'] == False:
        video_dict = compute.video_metadata(uploaded_file)
    
    if st.session_state['have_preds'] == False:
        preds = compute.predict(video_dict)

    if st.session_state['have_preprocessed'] == False:
        compute.postprocess_videos(video_dict, preds)

    ui.analysis_dashboard()