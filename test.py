import os
import csv
import cv2
import time
import pickle
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

from win32com.client import Dispatch
def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Load the pre-trained DNN model for face detection
model = 'data/deploy.prototxt'
weights = 'data/res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(model, weights)

video = cv2.VideoCapture(0)  # 0 for internal webcam, 1 for external

try:
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
except:
    print("Pickle files not found. Make sure you run the data capturing script first.")
    exit()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# image_bg=cv2.imread("background.png")

cols=['Names','Date','Time']

while True:
    ret, frame = video.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    
    # Preprocess the frame for the DNN model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()  # Perform face detection
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Crop and resize the image
            crop_img = frame[startY:endY, startX:endX]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            
            # Make prediction
            output = knn.predict(resized_img)
            
            ts=time.time()
            date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
            detected=[str(output[0]),str(date),str(timestamp)]
            exist=os.path.isfile("Detected/DetectionDetails" + date + ".csv")

            # Check if the predicted label is valid
            predicted_name = output[0]
            if predicted_name in LABELS:
                cv2.putText(frame, str(predicted_name), (startX, startY - 20), cv2.FONT_ITALIC, 1, (255, 150, 150), 2)
            else:
                cv2.putText(frame, "Unknown Person", (startX, startY - 20), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
    # Display the frame
    cv2.imshow("Face Detection", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) == ord('o'):
        speak("Details Logged") #Says OK when recognized face is logged into the data file
        if exist:
            with open("Detected/DetectionDetails" + date + ".csv" , '+a') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(detected)
            csvfile.close()
        else:
            with open("Detected/DetectionDetails" + date + ".csv" , '+a') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(cols)
                writer.writerow(detected)
            csvfile.close()
video.release()
cv2.destroyAllWindows()
