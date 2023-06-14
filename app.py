# Imports
import matplotlib
import streamlit as st
import sys

# Set matplotlib backend
matplotlib.use('WebAgg')

# Add project root to system path
sys.path.append('.')

# Import app modules
from streetstudy.streamlit_modules import compute
from streetstudy.streamlit_modules import state
from streetstudy.streamlit_modules import ui

# Configure Streamlit app
st.set_page_config(
    page_title='StreetStudy',
    page_icon='🚶🏽‍♀️🚶',
    layout='wide',
    menu_items={
        'Get Help': 'https://github.com/sardarchitect/ucsd_capstone',
        'Report a bug': 'https://www.arvinder.me',
        'About': 'StreetStudy is an experimental project by Arvinder Singh'
    }
)

# Initialize app state
state.initialize()

# Display UI header and get uploaded file
ui.header()
uploaded_file = ui.sidebar()

# Run analysis if app is in 'running' state
if st.session_state['running'] == True:
    video_dict = compute.get_video_metadata(uploaded_file)
    preds = compute.predict(video_dict)
    
    if not st.session_state.get('cache_postprocess'):
        compute.postprocess_videos(video_dict, preds)
        
    ui.analysis_dashboard(video_dict, preds)