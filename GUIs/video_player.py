from flask import Flask, render_template, Response
import pandas as pd
import cv2


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


@app.route('/')
def index():
    df = pd.read_csv(CSV_PATH)
    df[START_TIME_COL] = df[START_FRAME_COL] / VIDEO_FPS
    return render_template('index.html', video_path=VIDEO_NAME,
                           table=df, titles=df.columns.values, start_time_col_num=6)


#
# def generate():
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             ret, jpeg = cv2.imencode('.jpg', frame)
#             if ret:
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n'
#                        b'X-Duration: ' + str(duration).encode() + b'\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
#         else:
#             break
#     cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
