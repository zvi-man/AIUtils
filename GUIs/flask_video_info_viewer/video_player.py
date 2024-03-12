from flask import Flask, render_template, send_file, url_for
import pandas as pd
import os

from KMUtils.GUIs.flask_video_info_viewer.video_manager import VideoManager
from KMUtils.GUIs.flask_video_info_viewer.video_player_config import VideoPlayerConfig


app = Flask(__name__)


@app.route('/play_video/<video_name>')
def view_video(video_name: str):
    csv_path = os.path.join(VideoPlayerConfig.VIDEO_DIR, video_name, VideoPlayerConfig.CSV_NAME)
    df = pd.read_csv(csv_path)
    if VideoPlayerConfig.START_TIME_COL not in df.columns.values:
        df[VideoPlayerConfig.START_TIME_COL] = df[VideoPlayerConfig.START_FRAME_COL] / VideoPlayerConfig.VIDEO_FPS
    start_time_col_num = df.columns.get_loc(VideoPlayerConfig.START_TIME_COL)
    img_name_col_num = df.columns.get_loc(VideoPlayerConfig.IMG_NAME_COL)
    return render_template('view_video.html', video_name=video_name,
                           table=df, titles=df.columns.values, start_time_col_num=start_time_col_num,
                           img_name_col_num=img_name_col_num,
                           serve_image_url=url_for('serve_image', video_dir=video_name))


@app.route('/')
def index():
    video_dirs = [directory for directory in os.listdir(VideoPlayerConfig.VIDEO_DIR) if
                  os.path.isdir(os.path.join(VideoPlayerConfig.VIDEO_DIR, directory))]
    video_manager = VideoManager(VideoPlayerConfig.VIDEO_DIR)
    df = video_manager.get_video_info_df()
    video_name_col_num = df.columns.get_loc(VideoPlayerConfig.VIDEO_NAME_COL)
    return render_template('index.html', video_files=video_dirs, table=df,
                           titles=df.columns.values, video_name_col_num=video_name_col_num)


@app.route('/video/<video_dir>')
def serve_video(video_dir):
    return send_file(os.path.join(VideoPlayerConfig.VIDEO_DIR, video_dir, VideoPlayerConfig.VIDEO_NAME), mimetype='video/mp4')


@app.route('/image/<video_dir>')
@app.route('/image/<video_dir>/<img_name>')
def serve_image(video_dir: str, img_name: str = None):
    return send_file(os.path.join(VideoPlayerConfig.VIDEO_DIR, video_dir, img_name))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
