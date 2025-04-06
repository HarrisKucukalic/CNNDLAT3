from flask import Flask, render_template, Response
from camera import VideoCamera
from DogDetectorLive import LostMemeberDetector
# Source: https://www.youtube.com/watch?v=-4v4A550K3w
app = Flask(__name__)

# Home page = '/', about would be '/about', etc.
@app.route('/')

def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame
              + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # debug allows you to change code while running and can re-render live
    app.run(host='0.0.0.0', port='5000', debug=True)