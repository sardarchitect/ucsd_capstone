# Imports
import matplotlib
matplotlib.use('WebAgg')
import streamlit as st

# App Library
import sys
sys.path.append('.')
from streetstudy.streamlit_modules import compute
from streetstudy.streamlit_modules import state
from streetstudy.streamlit_modules import ui

# Configs
st.set_page_config(
    page_title="StreetStudy",
    page_icon="🚶🏽‍♀️🚶",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/sardarchitect/ucsd_capstone',
        'Report a bug': "https://www.arvinder.me",
        'About': "# StreetStudy is an experimental project by Arvinder Singh"
    }
)

state.initialize()

# UI
ui.header()
uploaded_file = ui.sidebar()
# Dashboard
if st.session_state['running'] == True:
    #TODO: save preds to cache for faster loading
    video_dict = compute.get_video_metadata(uploaded_file)
    preds = compute.predict(video_dict)
    if st.session_state['cache_postprocess'] == False:
        compute.postprocess_videos(video_dict, preds)
        
    ui.analysis_dashboard(video_dict, preds)