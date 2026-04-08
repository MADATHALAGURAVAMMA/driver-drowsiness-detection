import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame

# -----------------------------
# Initialize Alarm
# -----------------------------
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

def sound_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

# -----------------------------
# Eye Aspect Ratio Function
# -----------------------------
def eye_aspect_ratio(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# -----------------------------
# Threshold Values
# -----------------------------
EAR_THRESHOLD = 0.25
FRAME_CHECK = 20
counter = 0


# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_draw = mp.solutions.drawing_utils


# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640,480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS
            )

            h, w, _ = frame.shape

            # Left Eye Landmarks
            left_eye_points = [33,160,158,133,153,144]

            # Right Eye Landmarks
            right_eye_points = [362,385,387,263,373,380]

            left_eye = []
            right_eye = []

            for p in left_eye_points:
                x = int(face_landmarks.landmark[p].x * w)
                y = int(face_landmarks.landmark[p].y * h)
                left_eye.append((x,y))

            for p in right_eye_points:
                x = int(face_landmarks.landmark[p].x * w)
                y = int(face_landmarks.landmark[p].y * h)
                right_eye.append((x,y))


            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)

            ear = (ear_left + ear_right) / 2.0

            # Draw Eye Points
            for x,y in left_eye:
                cv2.circle(frame,(x,y),2,(0,255,0),-1)

            for x,y in right_eye:
                cv2.circle(frame,(x,y),2,(0,255,0),-1)


            # Drowsiness Detection
            if ear < EAR_THRESHOLD:

                counter += 1

                if counter >= FRAME_CHECK:

                    cv2.putText(frame,"DROWSINESS ALERT!",
                                (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,(0,0,255),3)

                    sound_alarm()

            else:
                counter = 0


            cv2.putText(frame,
                        f"EAR: {ear:.2f}",
                        (450,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(255,255,255),2)

    cv2.imshow("Driver Drowsiness Detection",frame)

    key = cv2.waitKey(1)

    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
