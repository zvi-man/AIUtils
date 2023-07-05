from flask import Flask, render_template, Response, request
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


def get_video_fps(video_path: str) -> float:
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.release()
    return fps


# Constants
VIDEO_PATH = r"D:\עבודה צבי\VehicleColorClassification\DataSets\HomeMade1\drive_through_raanana.mp4"
VIDEO_NAME = r"drive_through_raanana.mp4"
VIDEO_FPS = get_video_fps(VIDEO_PATH)
CSV_PATH = r"D:\עבודה צבי\LPRIL\KMUtils\GUIs\static\drive_through_raanana.csv"
START_FRAME_COL = "start_frame"
START_TIME_COL = "start_time"
app = Flask(__name__)


@app.route('/play_video')
def index(video_path: str = VIDEO_NAME, csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    df[START_TIME_COL] = df[START_FRAME_COL] / VIDEO_FPS
    start_time_col_num = df.columns.get_loc(START_TIME_COL)
    return render_template('index.html', video_path=video_path,
                           table=df, titles=df.columns.values, start_time_col_num=start_time_col_num)


app.config['UPLOAD_FOLDER'] = 'static/'


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['video_path']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        if not os.path.exists(video_path):
            f.save(video_path)
        f = request.files['csv_file_path']
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(csv_path)
        return index(video_path=video_path, csv_path=csv_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
