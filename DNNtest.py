# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os

#os.system("sudo modprobe bcm2835-v4l2")

'''CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]'''

CLASSES = ["", "", "", "", "",
    "", "", "", "", "", "", "",
    "", "", "", "person", "", "",
    "", "", ""]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
fps = FPS().start()

# loop over the frames from the video stream
count=0
while (True):
    frame = vs.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    #print(detections)

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if(confidence > 0.2):
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = CLASSES[idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            if(label=='person'):
                count+=1
                print("number of persons ",count)

    if(count == 0):
        print("No person detected")
    if(count == 1):
        print("1 person detected")
    if(count > 2):
        print("Additional person detected in the room.")
 
    # show the output frame
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    fps.update()
    count = 0
    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

    
