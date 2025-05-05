import base64

import cv2
from flask import Flask, render_template, Response, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from camera import VideoCamera
from LiveObjectDetector import LostMemeberDetector
from FaceReader import FaceDetector
from flask_socketio import SocketIO, emit

DOG_UPLOAD = r'C:\Users\Harris\PycharmProjects\CNNDLAT3\Final\dog_photo_upload'
HUMAN_UPLOAD = r'C:\Users\Harris\PycharmProjects\CNNDLAT3\Final\human_photo_upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jepg', 'gif'}
USERNAME = 'admin'
PASSWORD = 'secret'

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def requires_auth(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
def authenticate():
    return Response(
        'Login required.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Source: https://www.youtube.com/watch?v=-4v4A550K3w,
app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('signal')
def handle_signal(data):
    # Relay signaling messages between peers
    emit('signal', data, broadcast=True, include_self=False)

# Home page = '/', about would be '/about', etc.
@app.route('/')
@requires_auth
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/human_detection')
def human():
    return render_template('human.html',  human=False, face=False)

@app.route('/dog_detection')
def dog():
    return render_template('dog.html', human=False, face=False)

@app.route('/upload/human', methods=['GET', 'POST'])
def human_upload():
    if request.method == 'POST':
        person_file = request.files.get('person_file')
        face_file = request.files.get('face_file')
        if person_file and allowed_file(person_file.filename):
            npimg = np.frombuffer(person_file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            detector = LostMemeberDetector(human=True)
            processed_img = detector.get_image_prediction(img)

            # Convert to base64 to embed in HTML
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/jpeg;base64,{img_b64}"

            return render_template('human_results.html', image_uri=img_uri)
        if face_file and allowed_file(face_file.filename):
            npimg = np.frombuffer(face_file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            face_reader = FaceDetector()
            processed_img = face_reader.process_img(img)

            # Convert to base64 to embed in HTML
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/jpeg;base64,{img_b64}"

            return render_template('human_results.html', image_uri=img_uri)

    return render_template('human_upload.html')

@app.route('/upload/dog', methods=['GET', 'POST'])
def dog_upload():
    if request.method == 'POST':
        file = request.files.get('dog_file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            detector = LostMemeberDetector(human=False)
            processed_img = detector.get_image_prediction(img)

            # Convert to base64 to embed in HTML
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/jpeg;base64,{img_b64}"

            return render_template('dog_results.html', image_uri=img_uri)
    return render_template('dog_upload.html')

@app.route('/submit_location', methods=['POST'])
def submit_location():
    street = request.form['street']
    suburb = request.form['suburb']
    city = request.form['city']
    source = request.form.get('source')
    if source in ['human', 'dog']:
        return jsonify({"message": "Location uploaded successfully"})
    else:
        return jsonify({"error": "Unknown source"}), 400

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame
              + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    is_human = request.args.get('human', 'false').lower() == 'true'
    is_face = request.args.get('face', 'false').lower() == 'true'
    return Response(gen(VideoCamera(human=is_human, face=is_face)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # debug allows you to change code while running and can re-render live
    app.run(host='0.0.0.0', port='5000', debug=True)