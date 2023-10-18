import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile

# Constants
CSV_VIDEO_TIME = "start_frame"

# Function to read CSV file
def read_csv(file):
    df = pd.read_csv(file)
    return df


# Function to display video and object information
def display_video_with_info(video_file, csv_file):

    video_capture = cv2.VideoCapture(video_file.name)

    csv_data = read_csv(csv_file)

    st.video(video_file)


    time_slider = st.slider("Select Time in Video (seconds)", 0,
                            int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / video_capture.get(cv2.CAP_PROP_FPS), 1))

    # Find the object info at the selected time
    object_info = csv_data[(csv_data['Time_in_video'] <= time_slider) & (
                (csv_data['Time_in_video'] + csv_data['Duration']) >= time_slider)]

    if not object_info.empty:
        # Display CSV data as a table
        st.table(object_info)
    else:
        st.write("No object information available for the selected time.")


# Streamlit UI
st.title("Video and Object Info Viewer")

uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi"])
if uploaded_video:
    st.video(uploaded_video)

    uploaded_csv = st.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_csv:
        df = read_csv(uploaded_csv)
        table_view, table_control, image_view = st.tabs(["Table View", "Table Control", "Image View"])

        with table_view:
            st.header("Object Information")
            st.write(df, width=800)

        with table_control:
            st.header("Table Control")

        with image_view:
            st.header("Image View")





