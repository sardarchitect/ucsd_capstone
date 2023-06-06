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
        st.button(label=('Run' if st.session_state['is_run'] == False else "Stop"), on_click=state.set_run)
        # CITATIONS 
        st.subheader("Citations")
        return uploaded_file

def dashboard_form():
    with st.form("my_form"):
        st.write("Customize Output")
        frame_number = st.slider("slider_frame_number")
        plot_type = st.radio("radio_plot_type", options=["heatmap", "bounding_boxes", "directional_arrows"])

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            postprocess.show_plot(frame_number)
        