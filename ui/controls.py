import streamlit as st

def sidebar_controls():

    st.sidebar.header("⚙️ Settings")

    pop_size = st.sidebar.slider("Population", 10, 100, 30)
    sensors = st.sidebar.slider("Sensors", 5, 20, 10)
    generations = st.sidebar.slider("Generations", 10, 100, 30)

    return pop_size, sensors, generations