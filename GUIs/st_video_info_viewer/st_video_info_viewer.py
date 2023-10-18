import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile

# Constants
CSV_VIDEO_TIME_COL = "start_time"


class VideoInfoViewer:
    def __init__(self) -> None:
        self.init_session_state()
        self.init_main_window()

    @staticmethod
    def init_session_state():
        if 'start_time' not in st.session_state:
            st.session_state['start_time'] = 5

    def init_main_window(self):
        st.title("Video and Object Info Viewer")

        uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi"])

        if uploaded_video:
            default_val = st.session_state['start_time']
            st.write(default_val)
            placeholder = st.empty()
            placeholder.video(uploaded_video, start_time=default_val)

            st.button("Jump", on_click=self.set_video_time, args=(100,))

            uploaded_csv = st.file_uploader("Upload CSV File", type=['csv'])
            if uploaded_csv:
                df = self.read_csv(uploaded_csv)
                table_view, table_control, image_view = st.tabs(["Table View", "Table Control", "Image View"])

                with table_view:
                    st.header("Object Information")
                    st.write(df)

                with table_control:
                    st.header("Table Control")
                    self.plot_special_table(df)

                with image_view:
                    st.header("Image View")

    def plot_special_table(self, df: pd.DataFrame) -> None:
        columns = st.columns(len(df.columns))
        fields = df.columns
        for col, field_name in zip(columns, fields):
            # header
            col.write(field_name)

        for row_num, row in df.iterrows():
            columns = st.columns(len(df.columns))
            for i, (col_name, val) in enumerate(row.iteritems()):
                if col_name == CSV_VIDEO_TIME_COL:
                    button_hold = columns[i].empty()
                    button_hold.button(f"{val}", on_click=self.set_video_time, args=(val, ))
                else:
                    columns[i].write(val)

    @staticmethod
    def set_video_time(time: int):
        print(f"Setting time: {time}")
        st.session_state['start_time'] = time
        st.experimental_rerun()

    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


VideoInfoViewer()
