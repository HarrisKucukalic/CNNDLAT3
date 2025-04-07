from flask import Flask, render_template, Response, request, jsonify
import os
from werkzeug.utils import secure_filename
from camera import VideoCamera

DOG_UPLOAD = r'C:\Users\Harris\PycharmProjects\CNNDLAT3\Final\dog_photo_upload'
HUMAN_UPLOAD = r'C:\Users\Harris\PycharmProjects\CNNDLAT3\Final\human_photo_upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jepg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Source: https://www.youtube.com/watch?v=-4v4A550K3w,
app = Flask(__name__)

# Home page = '/', about would be '/about', etc.
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/human_detection')
def human():
    return render_template('human.html',  human=False, face=False)

@app.route('/dog_detection')
def dog():
    return render_template('dog.html', human=False, face=False)

@app.route('/upload/human', methods=['GET', 'POST'])
def human_upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(HUMAN_UPLOAD, filename))
            return 'Human photo uploaded successfully'
    return render_template('human_upload.html')

@app.route('/upload/dog', methods=['GET', 'POST'])
def dog_upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(DOG_UPLOAD, filename))
            return 'Dog photo uploaded successfully'
    return render_template('dog_upload.html')

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