# 🧠 Real-Time Facial Detection and Recognition System

A Python-based real-time facial recognition system using OpenCV's Deep Neural Network (DNN) module and a K-Nearest Neighbors (KNN) classifier. The system detects and recognizes human faces via webcam input and logs recognition data for potential use in attendance or access systems.

---

## 📌 Features

- 🔍 **Face Detection** using OpenCV’s DNN module (`res10_300x300_ssd_iter_140000_fp16.caffemodel`)
- 🧠 **Face Recognition** using K-Nearest Neighbors (KNN) algorithm
- 📸 Real-time **face data capture and labeling** for dynamic training
- 💾 Persistent data storage using `pickle` for names and face embeddings
- 🗣️ **Text-to-speech feedback** using `pywin32` for confirmation
- 📝 CSV-based **logging of recognized faces** with date and time stamps

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **OpenCV**
- **scikit-learn**
- **NumPy**
- **pickle**
- **pywin32 (for TTS on Windows)**

---

## 📂 Project Structure

📦project-root/
├── new_faces.py # Capture new face data for training
├── ttest.py # Real-time recognition and attendance logging
├── data/ # Stores DNN models, face data, and labels
│ ├── deploy.prototxt
│ ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
│ ├── faces_data.pkl # Stored face embeddings
│ └── names.pkl # Corresponding labels
├── Detected/ # Stores CSV attendance logs
│ └── DetectionDetailsDD-MM-YYYY.csv

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install opencv-python numpy scikit-learn pywin32
```

### 2. Add New Faces to Dataset
Run the script and follow the prompt to input a person's name:
```bash
python new_faces.py
```
  - Captures 100 face samples per person

  - Saves cropped and resized (50x50) grayscale face data to data/faces_data.pkl

  - Updates names in data/names.pkl

### 3. Start Real-Time Recognition
```bash
python ttest.py
```
  - Starts webcam, detects and recognizes faces
  
  - Press o to log the recognized person’s name, date, and time
  
  - Press q to quit the session

### 📈 Performance
  - Detection confidence threshold: > 0.5
  
  - Real-time performance: ~20 FPS
  
  - KNN classification latency: < 100ms per face

### ✅ Future Improvements
  - Add support for face re-training without restarting the app
  
  - Replace KNN with a deep learning-based classifier for higher accuracy
  
  - Add GUI for user-friendly management
  
  - Improve unknown face handling with threshold logic
  
  - Deployable attendance dashboard

### 👤 Author
    Pranjal Kundu
    Graphic Era Hill University, Dehradun
    “Always learning, always building.”

### 📃 License
    This project is open-source and available under the MIT License.
