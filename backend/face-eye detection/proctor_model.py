import cv2
import mediapipe as mp
import numpy as np
import math
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class InterviewProctor:
    def __init__(self):
        # 1. Automatically download the lightweight task file if it doesn't exist
        self.model_path = 'face_landmarker.task'
        if not os.path.exists(self.model_path):
            print("Downloading lightweight Face Landmarker model (2.6 MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete!")

        # 2. Setup the modern Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Eye Aspect Ratio (EAR) thresholds for Blink Detection
        self.EAR_THRESHOLD = 0.21
        self.blink_count = 0
        self.is_blinking = False

    def calculate_ear(self, eye_landmarks):
        """Calculates the Eye Aspect Ratio (EAR) based on Review 3 - Paper 4 methodology."""
        # Vertical distances
        v1 = math.dist(eye_landmarks[1], eye_landmarks[5])
        v2 = math.dist(eye_landmarks[2], eye_landmarks[4])
        # Horizontal distance
        h = math.dist(eye_landmarks[0], eye_landmarks[3])
        return (v1 + v2) / (2.0 * h)

    def analyze_frame(self, frame):
        # Convert BGR to RGB and package it into a MediaPipe Image object
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detection using the new Tasks API
        results = self.detector.detect(mp_image)
        
        h, w, _ = frame.shape
        status = "Focused"
        color = (0, 255, 0)
        
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # 1. EXTRACT LANDMARKS
                pts = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks]
                
                # Left and Right eye indices in standard Face Mesh
                left_eye = [pts[33], pts[160], pts[158], pts[133], pts[153], pts[144]]
                right_eye = [pts[362], pts[385], pts[387], pts[263], pts[373], pts[380]]
                
                # 2. BLINK DETECTION (EAR)
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < self.EAR_THRESHOLD:
                    if not self.is_blinking:
                        self.blink_count += 1
                        self.is_blinking = True
                else:
                    self.is_blinking = False

                # 3. HEAD POSE ESTIMATION (Cheat Detection)
                # Nose tip, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
                face_2d = np.array([pts[1], pts[152], pts[33], pts[263], pts[61], pts[291]], dtype=np.float64)
                face_3d = np.array([
                    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
                ])

                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, h/2], [0, focal_length, w/2], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                
                # FIX 1: Removed the *360 multiplier. OpenCV already outputs degrees.
                y_angle = angles[1]

                # FIX 2: Swapped the labels to correctly map to your mirrored webcam feed
                # 4. STATUS DETERMINATION
                if y_angle < -30:  # Changed from -10
                    status = "Looking Right (Warning)"
                    color = (0, 0, 255)
                elif y_angle > 30:   # Changed from 10
                    status = "Looking Left (Warning)"
                    color = (0, 0, 255)

                # Draw Visuals
                cv2.putText(frame, f"Candidate Focus: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Blinks Detected: {self.blink_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 255), 1)
                cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 255), 1)

        else:
            cv2.putText(frame, "No Candidate Detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

if __name__ == "__main__":
    print("Starting HireSense Proctoring Module...")
    cap = cv2.VideoCapture(0)
    proctor = InterviewProctor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror image
        processed_frame = proctor.analyze_frame(frame)
        
        cv2.imshow('HireSense Candidate Monitor', processed_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()