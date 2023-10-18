import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile

# Constants
CSV_VIDEO_TIME = "start_frame"


def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def plot_special_table(df: pd.DataFrame) -> None:
    st.write(df)


def set_video_time(time: int):
    st.session_state['start_time'] = time


if 'start_time' not in st.session_state:
    st.session_state['start_time'] = 5

st.title("Video and Object Info Viewer")

uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi"])


if uploaded_video:
    placeholder = st.empty()
    default_val = st.session_state['start_time']
    placeholder.video(uploaded_video, start_time=default_val)

    st.button("Jump", on_click=set_video_time, args=(100, ))

    uploaded_csv = st.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_csv:
        df = read_csv(uploaded_csv)
        table_view, table_control, image_view = st.tabs(["Table View", "Table Control", "Image View"])

        with table_view:
            st.header("Object Information")
            st.write(df)

        with table_control:
            st.header("Table Control")
            plot_special_table(df)

        with image_view:
            st.header("Image View")





