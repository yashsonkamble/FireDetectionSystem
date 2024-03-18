# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request, Response
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import cv2    #opencv-python
from playsound import playsound

CLASSES = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "person", "", "", "", "", ""]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('camera'))
    return render_template('login.html', error=error)


@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template('camera.html')


def get_frame():
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    fps = FPS().start()
    # loop over the frames from the video stream
    count = 0
    while (True):
        frame = vs.read()

        fire = fire_cascade.detectMultiScale(frame, 1.2, 5)
        if (fire != ()):
            print('fire detected')
            cv2.putText(frame, "Fire Detected", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            playsound('voice.mp3')

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        # print(detections)

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence > 0.2):
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = CLASSES[idx]
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                if (label == 'person'):
                    count += 1
                    print("number of persons ", count)
        cv2.putText(frame, "Persons detected :" + str(count), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        fps.update()
        count = 0

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')


@app.route('/video_stream')
def video_stream():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
    