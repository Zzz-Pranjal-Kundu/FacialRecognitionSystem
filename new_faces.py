import cv2
import pickle
import numpy as np
import os

# Load the pre-trained DNN model for face detection
model = 'data/deploy.prototxt'
weights = 'data/res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(model, weights)

video = cv2.VideoCapture(0)  # 0 for internal webcam, 1 for external

face_data=[]
name=input("Enter the name: ")
i=0

while True:
    ret, frame = video.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    # Preprocess the frame for the DNN model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward() # Perform face detection
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            # Draw the bounding box
            label = f"Confidence: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #crop image
            crop_img = frame[startY:endY, startX:endX]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(face_data)<=100 and i%10==0:
                face_data.append(resized_img)
                cv2.putText(frame, str(len(face_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
            i+=1

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data)==100:
        break
video.release()
cv2.destroyAllWindows()

face_data=np.asarray(face_data).reshape(100, -1)

#create names.pkl if not available else append the new data into it
if 'names.pkl' not in os.listdir('data/'):
    with open('data/names.pkl', 'wb')  as f:
        names=[name]*100
        pickle.dump(names,f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names,f)

#create faces_data.pkl if not available else append the new data into it
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data,f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        face=pickle.load(f)
    face=np.append(face,face_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face,f)
