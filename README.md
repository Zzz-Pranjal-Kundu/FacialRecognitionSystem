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

