import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile
import os

# Constants
VIDEO_TIME_COL = "start_time"
IMAGE_PATH_COL = "img_path"


class VideoInfoViewerBackEnd(object):
    DATA_PATH = "/home/zvi/Projects/LPRIL/KMUtils/GUIs/st_video_info_viewer/DATA"
    CSV_NAME = "data.csv"
    VIDEO_NAME = "video.mp4"

    @classmethod
    def get_all_dirs(cls):
        return os.listdir(cls.DATA_PATH)

    @classmethod
    def get_video_path(cls, video: str) -> str:
        return os.path.join(cls.DATA_PATH, video, cls.VIDEO_NAME)

    @classmethod
    def get_csv_path(cls, video: str) -> str:
        return os.path.join(cls.DATA_PATH, video, cls.CSV_NAME)


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
        video = st.selectbox("Select Video", VideoInfoViewerBackEnd.get_all_dirs())

        default_val = st.session_state['start_time']
        placeholder = st.empty()
        placeholder.video(VideoInfoViewerBackEnd.get_video_path(video), start_time=default_val)
        num = st.number_input("set video time sec", value=default_val, step=1, key="num_in1")
        st.button("Jump", on_click=self.set_video_time, args=(num,), key="but1")

        df = self.read_csv(VideoInfoViewerBackEnd.get_csv_path(video))
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
                if col_name == VIDEO_TIME_COL:
                    button_hold = columns[i].empty()
                    button_hold.button(f"{val}", on_click=self.set_video_time, args=(val, ), key="but2")
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
