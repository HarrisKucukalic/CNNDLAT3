import base64
import cv2
from flask import Flask, render_template, Response, request, jsonify
import os
import numpy as np
from camera import VideoCamera
from LiveObjectDetector import LostMemeberDetector
from FaceReader import FaceDetector
from flask_socketio import SocketIO, emit
import face_recognition
import pickle
import csv
from functools import wraps

# Set our "databases" for lost people and pets - currently CSV files to show proof of concept.
LOST_HUMAN_CSV = r'C:\projects\CNNDLAT3\TrackMate\lost_databases\lost_human_members.csv'
LOST_PET_CSV = r'C:\projects\CNNDLAT3\TrackMate\lost_databases\lost_pet_members.csv'

# These are the image types allowed to be uploaded. They are kept limited for now to ensure higher security and avoid the uploading of unusual file types.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Password to access the GUI on the web, to ensure security.
USERNAME = 'admin'
PASSWORD = 'secret'

# Source: https://www.youtube.com/watch?v=-4v4A550K3w
app = Flask(__name__)
socketio = SocketIO(app)


# This is the authemtication checker, ensures that the password and username is correct
def check_auth(username, password):
    return username == USERNAME and password == PASSWORD


# This function is essentially a wrapper for the check_auth function
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # gets username and password from web app
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            # If incorrect, returns an unauthorised response and makes user re-enter username and password. If cancelled, returns 401 error.
            return authenticate()
        # If authentication successful, continue with the function being authenticated
        return f(*args, **kwargs)

    return decorated


def authenticate():
    return Response(
        'Login required.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )


# Used to check the file extension and see if it's in our dictionary of accepted file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@socketio.on('signal')
def handle_signal(data):
    # Relay signaling messages between peers
    emit('signal', data, broadcast=True, include_self=False)


# Home page = '/', about would be '/about', etc.
@app.route('/')
@requires_auth
def index():
    return render_template('index.html')


@app.route('/lost_table')
def lost_table():
    lost_humans = []
    lost_pets = []
    # Creates our lost human and pets table for our lost & found
    try:
        with open(LOST_HUMAN_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lost_humans.append({
                    'type': 'Human',
                    'name': row['name'],
                    'street': row['street'],
                    'suburb': row['suburb'],
                    'city': row['city']
                })
    except FileNotFoundError:
        pass

    try:
        with open(LOST_PET_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lost_pets.append({
                    'type': 'Pet',
                    'name': row['name'],
                    'street': row['street'],
                    'suburb': row['suburb'],
                    'city': row['city']
                })
    except FileNotFoundError:
        pass

    return render_template('lost_table.html', lost_humans=lost_humans, lost_pets=lost_pets)


@app.route('/already_lost')
def already_lost():
    return render_template('already_lost.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/already_lost/human_detection')
def human():
    return render_template('human.html', human=True, face=False)


@app.route('/already_lost/dog_detection')
def dog():
    return render_template('dog.html', human=False, face=False)


@app.route('/already_lost/upload/human', methods=['GET', 'POST'])
def human_upload():
    if request.method == 'POST':
        # used to choose weather the FaceDetector or LostMemberDetector will be used. Gets POST data from HTML page
        person_file = request.files.get('person_file')
        face_file = request.files.get('face_file')
        if person_file and allowed_file(person_file.filename):
            # Turns image into a readable numpy array that is fed into
            npimg = np.frombuffer(person_file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            detector = LostMemeberDetector(human=True)
            processed_img = detector.get_image_prediction(img)

            # Convert to base64 to display and embed in the HTML page
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/jpeg;base64,{img_b64}"

            return render_template('human_results.html', image_uri=img_uri)
        if face_file and allowed_file(face_file.filename):
            npimg = np.frombuffer(face_file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            face_reader = FaceDetector()
            processed_img = face_reader.process_img(img)

            # Convert to base64 to display and embed in the HTML page
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/jpeg;base64,{img_b64}"

            return render_template('human_results.html', image_uri=img_uri)

    return render_template('human_upload.html')


@app.route('/already_lost/upload/dog', methods=['GET', 'POST'])
def dog_upload():
    if request.method == 'POST':
        # Same as the human upload, but sets the human=False for the LostMemberDetector
        file = request.files.get('dog_file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            detector = LostMemeberDetector(human=False)
            processed_img = detector.get_image_prediction(img)

            # Convert to base64 to display and embed in the HTML page
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/jpeg;base64,{img_b64}"

            return render_template('dog_results.html', image_uri=img_uri)
    return render_template('dog_upload.html')


@app.route('/submit_location', methods=['POST'])
def submit_location():
    # This is where the inputted data for the location submission goes.
    # It is split into the human or pet csv, depending on which page the data was submitted from.
    street = request.form['street']
    suburb = request.form['suburb']
    city = request.form['city']
    source = request.form.get('source')
    # We hope to add a portion here that can get the predicted label and add it to the csv database.
    if not all([street, suburb, city, source]):
        return jsonify({"error": "Missing required fields"}), 400
    if source == 'new_human':
        filename = LOST_HUMAN_CSV
        name = request.form['new_human_name']
    elif source == 'new_pet':
        filename = LOST_PET_CSV
        name = request.form['new_dog']
    else:
        return jsonify({"error": "Unknown source"}), 400
    # Ensures the rows have the same format and used to create the csv if it doesn't exist in the directory.
    row = [name, street, suburb, city]
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['name', 'street', 'suburb', 'city'])
        writer.writerow(row)

    return jsonify({"message": f"{'Lost' if source.startswith('new_') else 'Found'} location uploaded successfully"})


@app.route('/new_lost')
def new_lost():
    return render_template('new_lost.html')


@app.route('/new_lost/new_dog', methods=['GET', 'POST'])
def new_dog():
    if request.method == 'POST':
        file = request.files.get('dog_file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            # This is used to check if a dog is detectable when a new pet is
            # being added to the lost dataset, checking if the predicted label is correct.
            detector = LostMemeberDetector(human=False)
            processed_img = detector.get_image_prediction(img)

            # Convert to base64 to display and embed in the HTML page
            _, buffer = cv2.imencode('.jpg', processed_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/jpeg;base64,{img_b64}"

            return render_template('new_dog_results.html', image_uri=img_uri)
    return render_template('new_dog.html')


@app.route('/new_lost/new_human', methods=['GET', 'POST'])
def new_human():
    face_file = request.files.get('face_file')
    # Sam concept as new_dog, but uses the face_recognition
    # and YOLOv12n baseline model for face and body detection.
    if face_file and allowed_file(face_file.filename):
        name = os.path.splitext(face_file.filename)[0]
        npimg = np.frombuffer(face_file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)
        if len(face_encodings) == 0:
            return "No face detected. Please upload a clear image."

        new_encoding = face_encodings[0]

        # Load existing encodings from pickle file
        if os.path.exists("EncodeFile.p"):
            with open("EncodeFile.p", 'rb') as file:
                encoded_list_known, ids = pickle.load(file)
        else:
            encoded_list_known, ids = [], []

        # Append new encoding and ID to the pickle file
        encoded_list_known.append(new_encoding)
        ids.append(name)
        encoded_list_known_w_ids = [encoded_list_known, ids]
        # Save back to pickle file
        pkl_file = open("EncodeFile.p", 'wb')
        pickle.dump(encoded_list_known_w_ids, pkl_file)
        pkl_file.close()
        face_reader = FaceDetector()
        processed_img = face_reader.process_img(img)
        # Display image back to user in the web app.
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        img_uri = f"data:image/jpeg;base64,{img_b64}"

        return render_template('new_human_results.html', image_uri=img_uri, name=name)

    return render_template('new_human.html')


# This function is used to help stream data from a camera to the web app.
# Also from https://www.youtube.com/watch?v=-4v4A550K3w

camera = VideoCamera(human=True, face=False)


def gen(cam):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame
               + b'\r\n\r\n')


# This function returns the live camera feed to the web app, using the gen(camera) function from above.
# Also from https://www.youtube.com/watch?v=-4v4A550K3w
@app.route('/video_feed')
def video_feed():
    is_human = request.args.get('human', 'false').lower() == 'true'
    is_face = request.args.get('face', 'false').lower() == 'true'
    camera.set_mode(human=is_human, face=is_face)
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # debug allows you to change code while running and can re-render live
    app.run(host='0.0.0.0', port='5001', debug=True)
