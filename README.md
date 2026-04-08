# Driver Drowsiness Detection System

## Project Overview

Driver Drowsiness Detection is a safety system that monitors a driver in real-time and detects signs of fatigue or sleepiness. The system uses a webcam to capture the driver’s face and applies computer vision and deep learning techniques to determine whether the driver is alert or drowsy.

If drowsiness is detected, the system triggers an alarm sound to alert the driver and help prevent accidents.

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow
* Keras
* CNN (Convolutional Neural Network)

---

## System Workflow

1. Webcam captures the driver's face.
2. OpenCV processes video frames.
3. MediaPipe detects facial landmarks.
4. EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) are calculated.
5. CNN model predicts whether the driver is alert or drowsy.
6. If drowsiness is detected, an alarm sound is triggered.

---

## Project Structure

driver-drowsiness-detection
│
├── dataset
│   ├── train
│   │   ├── alert
│   │   └── drowsy
│   └── test
│       ├── alert
│       └── drowsy
│
├── train_model.py
├── detect_drowsiness.py
├── drowsiness_model.h5
├── alarm.wav
├── requirements.txt
└── README.md

---

## Installation

### Step 1: Clone the Repository

git clone https://github.com/your-username/driver-drowsiness-detection.git

### Step 2: Navigate to the Project Folder

cd driver-drowsiness-detection

### Step 3: Install Dependencies

pip install -r requirements.txt

---

## Training the Model

Run the following command to train the CNN model:

python train_model.py

After training, the model file **drowsiness_model.h5** will be created.

---

## Running the Detection System

To start real-time driver monitoring, run:

python detect_drowsiness.py

The webcam will start and the system will monitor the driver's eyes and mouth movements.

---

## Dataset

The model is trained using driver drowsiness image datasets containing two classes:

* Alert (Eyes open)
* Drowsy (Eyes closed)

Example datasets include:

* NTHU Driver Drowsiness Dataset
* Kaggle Driver Drowsiness Dataset

---

## Features

* Real-time driver monitoring
* Eye blink detection
* Yawning detection
* CNN-based drowsiness classification
* Alarm alert system

---

## Applications

* Smart vehicles
* Driver safety monitoring
* Accident prevention systems
* Transportation and logistics safety

---

## Future Enhancements

* Mobile application integration
* IoT-based driver monitoring
* Improved deep learning models
* Cloud-based monitoring systems

---

## Conclusion

This project demonstrates how computer vision and deep learning can be used to detect driver fatigue and improve road safety. By monitoring eye and mouth movements, the system can alert drivers before dangerous situations occur.

---

## Author

Project developed as part of MCA academic project.
