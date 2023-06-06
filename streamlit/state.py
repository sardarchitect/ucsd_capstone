import streamlit as st

def initialize():
    
    if 'is_run' not in st.session_state:
        st.session_state['is_run'] = False
        print("is_run:", st.session_state['is_run'])

def set_run():
    st.session_state['is_run'] = not(st.session_state['is_run'])
    print("is_run:", st.session_state['is_run'])