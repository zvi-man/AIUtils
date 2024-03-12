from dataclasses import dataclass
from typing import List
import os
import os.path as osp
import cv2
import pandas as pd

from KMUtils.GUIs.flask_video_info_viewer.video_player_config import VideoPlayerConfig


@dataclass
class VideoInfo:
    video_name: str
    has_csv: bool
    has_video: bool
    num_images: int
    video_length_sec: int
    additional_attributes: str


class VideoManager(object):
    def __init__(self, video_dir_path: str):
        self.video_dir_path = video_dir_path
        self.video_dir_paths = self._get_video_names(self.video_dir_path)
        self.video_info_list = []
        for video_path in self.video_dir_paths:
            video_info = self._get_video_info(video_path)
            self.video_info_list.append(video_info)

    @staticmethod
    def _get_video_fps(video_path: str) -> float:
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()
        return fps

    @staticmethod
    def _get_video_names(video_dir_path: str) -> List[str]:
        subdirs_with_images = []
        # Iterate over all subdirectories
        for root, dirs, files in os.walk(video_dir_path):
            # Check if any files in the current directory are images
            if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) for file in files):
                subdirs_with_images.append(root)
        return subdirs_with_images

    def _get_video_info(self, video_path: str) -> VideoInfo:
        video_name = osp.basename(video_path)
        has_csv = osp.exists(osp.join(video_path, VideoPlayerConfig.CSV_NAME))
        has_video = osp.exists(osp.join(video_path, VideoPlayerConfig.VIDEO_NAME))
        df = pd.read_csv(osp.join(video_path, VideoPlayerConfig.CSV_NAME))
        num_images = len(df)
        video_length_sec = self._get_video_length_sec(osp.join(video_path, VideoPlayerConfig.VIDEO_NAME))
        additional_attributes = ",".join([att for att in df.columns.values if att in VideoPlayerConfig.ATTRIBUTES])
        return VideoInfo(video_name, has_csv, has_video, num_images, video_length_sec, additional_attributes)

    @staticmethod
    def _get_video_length_sec(video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        duration_sec = total_frames / frame_rate
        cap.release()
        return int(duration_sec)

    def get_video_info_df(self) -> pd.DataFrame:
        df = pd.DataFrame([vars(video_info) for video_info in self.video_info_list])
        return df


if __name__ == '__main__':
    video_manager = VideoManager(VideoPlayerConfig.VIDEO_DIR)
    video_manager_df = video_manager.get_video_info_df()
    print(video_manager_df)
