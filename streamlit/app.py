import os
import streamlit as st
import inference

def set_session_on():
    st.session_state['start'] = True
    

def render_plot(option, video_dict, preds):
    st.session_state['render'] = True
    inference.plot_interactive(option, video_dict, preds)

if 'start' not in st.session_state:
    st.session_state.start = False

if 'render' not in st.session_state:
    st.session_state.render = False

st.set_page_config(
    page_title="StreetStudy",
    page_icon="🏙️",
    layout="wide",
)

# TITLE
st.title("🚶🏽‍♀️🚶🚶🏿‍♀️ StreetStudy 🚶🏿‍♂️🚶🏼‍♀️🚶🏽")
st.subheader("Analyze Pedestrian Traffic Using YOLOv5")

# SUMMARY
st.markdown("As urban areas become increasingly populated, architects and planners face the challenge of designing spaces that can accommodate high volumes of foot traffic. Furthermore, urban spaces are used in ways usually not intended by the designers. For instance, pedestrians may choose a shortcut through the grass rather than following the paved pathway provided by the designer if it is more efficient. The unique ways in which people may use public spaces is usually not understood well by the designers.")
st.markdown("By optimizing our environment to behave in ways that are more attuned to pedestrian needs, we can create better spaces for all. Moreover, this optimization can allow for more nature to flourish in our cities. For example, by finding patterns in the paths pedestrians take, designers can reduce the need for large slabs of concrete that inhibit biodiversity which would have thrived in its place.")
st.markdown("Redesigning public spaces with pedestrian needs in mind offers an opportunity to create sustainable, attractive urban areas that promote the societal and environmental enrichment.")
st.markdown("While there are existing products that provide analysis of foot traffic across, most seem to either use private mobile data to get an aggregation of street usage, or track customer density within controlled, indoor environments like grocery stores. Having a computer vision model that analyses pedestrian traffic on specific sections of an outdoor environment, for example a plaza, will provide designers a granular understanding of current space usage.")
st.divider()

# USAGE
with st.sidebar:
    st.subheader("Usage")
    st.markdown("1. Upload a .mp4 video (maximum 500mb in size) \
                \n 2. Press start \
                \n 3. The model results will be displayed below")

    # CITATIONS 
    st.subheader("Citations")

# VIDEO UPLOADER
uploaded_file = st.file_uploader(label='Upload a video to perform analysis')

start = st.button('Start', on_click=set_session_on)

if st.session_state['start'] == True:
    save_path = '.cache/'
    video_dict, preds = inference.pipeline(uploaded_file, save_path)

    # DASHBOARD
    st.video(os.path.join(save_path,"bbox/output.mp4"))

    met_col_1, met_col_2, met_col_3 = st.columns(3)
    with met_col_1:
        st.metric("Approximate Number of Unique People", 42)
    with met_col_2:
        st.metric("Approximate", 69)
    with met_col_3:
        st.metric("Approximate", 00)

if st.session_state['start'] == True:
    dash_col_1, dash_col_2 = st.columns(2)
    with dash_col_1:
        st.subheader("Pedestrian Density")
        st.video(os.path.join(save_path,"heatmap/output.mp4"))
    with dash_col_2:
        st.subheader("Pedestrian Flow")
        st.video(os.path.join(save_path,"arrows/output.mp4"))

if st.session_state['start'] == True:
    with st.form("my_form"):
        option = st.radio("Select Plot", ("Heatmap", "Directional Flow"))
        submitted = st.form_submit_button("Render", on_click=render_plot(option, video_dict, preds))