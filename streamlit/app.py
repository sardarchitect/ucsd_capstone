import os
import streamlit as st
import compute
import state
import ui_modules
import numpy as np
import plotly as plt

from streetstudy.common import utils

save_path = '.data_cache/'
utils.make_dir(save_path)

st.set_page_config(
    page_title="StreetStudy",
    page_icon="🏙️",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.arvinder.me',
        'Report a bug': "https://www.arvinder.me/contact",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# INITIALIZE
state.initialize()
    
# HEADER
ui_modules.header()

# SIDEBAR
uploaded_file = ui_modules.sidebar()

# DASHBOARD
if st.session_state['is_run'] == True:
    ui_modules.dashboard_form()
    
    video_dict = compute.video_metadata(uploaded_file)
    preds = compute.predict(video_dict, save_path)
    compute.postprocess_videos(video_dict, preds, save_path)

    st.video(os.path.join(save_path,"bbox/output.mp4"))
    dash_col_1, dash_col_2 = st.columns(2)
    with dash_col_1:
        st.subheader("Pedestrian Density")
        st.video(os.path.join(save_path,"heatmap/output.mp4"))
    with dash_col_2:
        st.subheader("Pedestrian Flow")
        st.video(os.path.join(save_path,"arrows/output.mp4"))