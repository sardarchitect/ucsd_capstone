import streamlit as st
import inference

# TITLE
st.title("StreetStudy")
st.subheader("Analyze Pedestrian Traffic Using YOLOv5")
st.divider()

# VIDEO UPLOADER
uploaded_file = st.file_uploader(label='Upload a video to perform analysis')
if st.button('Start'):
    inference.pipeline(uploaded_file)
    
    # DASHBOARD
    st.video("./outputs/bbox/output.mp4")
    st.video("./outputs/heatmap/output.mp4")
st.divider()

# SUMMARY
st.markdown("As urban areas become increasingly populated, architects and planners face the challenge of designing spaces that can accommodate high volumes of foot traffic. Furthermore, urban spaces are used in ways usually not intended by the designers. For instance, pedestrians may choose a shortcut through the grass rather than following the paved pathway provided by the designer if it is more efficient. The unique ways in which people may use public spaces is usually not understood well by the designers.")
st.markdown("By optimizing our environment to behave in ways that are more attuned to pedestrian needs, we can create better spaces for all. Moreover, this optimization can allow for more nature to flourish in our cities. For example, by finding patterns in the paths pedestrians take, designers can reduce the need for large slabs of concrete that inhibit biodiversity which would have thrived in its place.")
st.markdown("Redesigning public spaces with pedestrian needs in mind offers an opportunity to create sustainable, attractive urban areas that promote the societal and environmental enrichment.")
st.markdown("While there are existing products that provide analysis of foot traffic across, most seem to either use private mobile data to get an aggregation of street usage, or track customer density within controlled, indoor environments like grocery stores. Having a computer vision model that analyses pedestrian traffic on specific sections of an outdoor environment, for example a plaza, will provide designers a granular understanding of current space usage.")

# USAGE
st.subheader("Usage")
st.markdown("1. Upload a .mp4 video (maximum 500mb in size) \
            \n 2. Press start \
            \n 3. The model results will be displayed below")

# CITATIONS 
st.subheader("Citations")