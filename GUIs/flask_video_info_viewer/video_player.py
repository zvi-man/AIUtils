from flask import Flask, render_template, send_file, url_for
import pandas as pd
import cv2
import os

# TODO:
#  1. Add option to search the table
#  2. If the table is too long then scroll through table without loosing video
#  3. Add option to color the progress bar according to some data in table
#  V2 - display bounding box
#  Why do i need this version? well, i need an application to jump to locations in video,
#  Plus we need to show the user what we meant.
#  1. change video player to cv2
#  2. Add pause play and stop
#  3. Add progress bar
#  4.

# Constants
VIDEO_DIR = r"/home/zvi/Desktop/Videos"
CSV_NAME = "data.csv"
START_FRAME_COL = "start_frame"
START_TIME_COL = "start_time"
IMG_NAME_COL = "img_path"
VIDEO_NAME = "video.mp4"
VIDEO_FPS = 30
app = Flask(__name__)


def get_video_fps(video_path: str) -> float:
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.release()
    return fps


@app.route('/play_video/<video_name>')
def view_video(video_name: str):
    csv_path = os.path.join(VIDEO_DIR, video_name, CSV_NAME)
    df = pd.read_csv(csv_path)
    if START_TIME_COL not in df.columns.values:
        df[START_TIME_COL] = df[START_FRAME_COL] / VIDEO_FPS
    start_time_col_num = df.columns.get_loc(START_TIME_COL)
    img_name_col_num = df.columns.get_loc(IMG_NAME_COL)
    return render_template('view_video.html', video_name=video_name,
                           table=df, titles=df.columns.values, start_time_col_num=start_time_col_num,
                           img_name_col_num=img_name_col_num,
                           serve_image_url=url_for('serve_image', video_dir=video_name))


@app.route('/')
def index():
    video_dirs = [directory for directory in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, directory))]
    return render_template('index.html', video_files=video_dirs)


@app.route('/video/<video_dir>')
def serve_video(video_dir):
    return send_file(os.path.join(VIDEO_DIR, video_dir, VIDEO_NAME), mimetype='video/mp4')


@app.route('/image/<video_dir>')
@app.route('/image/<video_dir>/<img_name>')
def serve_image(video_dir: str, img_name: str = None):
    return send_file(os.path.join(VIDEO_DIR, video_dir, img_name))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
