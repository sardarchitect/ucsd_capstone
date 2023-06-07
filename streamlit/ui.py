import os
import streamlit as st
import state
import postprocess

def header():
    st.title("🚶🏽‍♀️🚶🚶🏿‍♀️ StreetStudy 🚶🏿‍♂️🚶🏼‍♀️🚶🏽")
    st.subheader("Analyze Pedestrian Traffic Using YOLOv5")
    st.markdown("As urban areas become increasingly populated, architects and planners face the challenge of designing spaces that can accommodate high\
             volumes of foot traffic. Furthermore, urban spaces are used in ways usually not intended by the designers. For instance, pedestrians may\
             choose a shortcut through the grass rather than following the paved pathway provided by the designer if it is more efficient.\
             The unique ways in which people may use public spaces is usually not understood well by the designers.")
    st.divider()
    
def sidebar():
    with st.sidebar:
        st.subheader("Usage")
        st.markdown("1. Upload a .mp4 video (maximum 500mb in size) \
                    \n 2. Press start \
                    \n 3. The model results will be displayed below")
        uploaded_file = st.file_uploader(label='Upload a video to perform analysis')
        st.button(label=('Run' if st.session_state['is_run'] == False else "Stop"), args=[uploaded_file], on_click=state.set_run)
        # CITATIONS 
        st.subheader("Citations")
        return uploaded_file

def dashboard():
    with st.form("dashboard_form"):
        form_col_1, form_col_2 = st.columns(2)
        st.write("Analysis Options")
        with form_col_1:
            display_type = st.radio("Display Type", options=["video", "interactive_plot"])
        with form_col_2:
            analysis_type = st.radio("Analysis Type", options=["heatmap", "bounding_boxes", "directional_arrows"])

        # Every form must have a submit button.
        
        if st.form_submit_button("Render"):
            st.session_state["display_type"] = display_type
            st.session_state["analysis_type"] = analysis_type
        plot()

def plot():
    if st.session_state["display_type"] == "video":
        st.video(os.path.join(st.session_state["save_path"], st.session_state["analysis_type"], "output.mp4"))
    if st.session_state["display_type"] == "interactive_plot":
        postprocess.show_plot(50)