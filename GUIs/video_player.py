from flask import Flask, render_template, Response
import cv2

# Constants
VIDEO_PATH = r"D:\עבודה צבי\VehicleColorClassification\DataSets\HomeMade1\drive_through_raanana.mp4"
CSV_PATH = r""
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def generate():
    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        ret, frame = cap.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            break
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
