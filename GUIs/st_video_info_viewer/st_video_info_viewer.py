import math
from typing import List

import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile
import os

# Constants
VIDEO_TIME_COL = "start_time"
IMAGE_PATH_COL = "img_path"
ID_COL = "ID"


class VideoInfoViewerBackEnd(object):
    DATA_PATH = "/home/zvi/Projects/LPRIL/KMUtils/GUIs/st_video_info_viewer/DATA"
    CSV_NAME = "data.csv"
    VIDEO_NAME = "video.mp4"

    @classmethod
    def get_all_dirs(cls):
        dir_list = os.listdir(cls.DATA_PATH)
        return cls.verify_dir_list(dir_list)

    @classmethod
    def get_video_path(cls, video: str) -> str:
        return os.path.join(cls.DATA_PATH, video, cls.VIDEO_NAME)

    @classmethod
    def get_csv_path(cls, video: str) -> str:
        return os.path.join(cls.DATA_PATH, video, cls.CSV_NAME)

    @classmethod
    def verify_dir_list(cls, dir_list: List[str]) -> List[str]:
        # TODO: Verify all files in dir
        return dir_list

    @classmethod
    def get_img_path(cls, df: pd.DataFrame, video_name: str, image_idx: int) -> str:
        image_name = df.iloc[image_idx][IMAGE_PATH_COL]
        return os.path.join(cls.DATA_PATH, video_name, image_name)


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
        video_name = st.selectbox("Select Video", VideoInfoViewerBackEnd.get_all_dirs())

        default_val = st.session_state['start_time']
        placeholder = st.empty()
        placeholder.video(VideoInfoViewerBackEnd.get_video_path(video_name), start_time=default_val)
        num = st.number_input("set video time sec", value=default_val, step=1, key="num_in1")
        st.button("Jump", on_click=self.set_video_time, args=(num,), key="but1")

        df = self.read_csv(VideoInfoViewerBackEnd.get_csv_path(video_name))
        table_view, table_control, image_view = st.tabs(["Table View", "Table Control", "Image View"])

        with table_view:
            st.header("Object Information")
            st.write(df)

        with table_control:
            st.header("Table Control")
            self.plot_special_table(df)

        with image_view:
            st.header("Image View")
            self.plot_images(df, video_name)

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

    @staticmethod
    def plot_images(df: pd.DataFrame, video_name: str) -> None:
        num_images_per_row = st.number_input("Number of Images per Row", min_value=1, value=5, step=1)

        num_images = df.shape[0]
        num_rows = math.ceil(8 / 5)

        for row in range(num_rows):
            cols = st.columns(num_images_per_row)
            for col_idx, col in enumerate(cols):
                image_idx = row * num_images_per_row + col_idx
                if image_idx < num_images:
                    image_path = VideoInfoViewerBackEnd.get_img_path(df, video_name, image_idx)
                    col.image(image_path, caption=f"ID: {df.iloc[image_idx][ID_COL]}", use_column_width=True)


VideoInfoViewer()
