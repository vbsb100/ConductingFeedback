#Make the window a little larger for the videos when we make a cohesive file
import cv2
import numpy as np
import mediapipe as mp
import pickle
from scipy.signal import find_peaks
import numpy as np
lwrist_y = []
lwrist_x = []
rwrist_x = []
rwrist_y = []
right_shoulder_x = []
left_shoulder_x = []
frame_count = []
frame_rate = 29.98
frame_num = 0

print("What is the video file name?")
name_vid = input()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For video input
cap = cv2.VideoCapture("./videos/" + name_vid + ".MOV")
fps =  cap.get(cv2.CAP_PROP_FPS)
# Resize the image while preserving aspect ratio
scale_percent = 80 # percent of original size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent / 100)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent / 100)
out = cv2.VideoWriter('./output/' + name_vid + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
 
    
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
    if results.pose_landmarks:
        left_shoulder_x = results.pose_landmarks.landmark[11].x
        right_shoulder_x = results.pose_landmarks.landmark[12].x
        lwrist_x = results.pose_landmarks.landmark[16].x
        lwrist_y = results.pose_landmarks.landmark[16].y
        rwrist_x = results.pose_landmarks.landmark[15].x
        rwrist_y = results.pose_landmarks.landmark[15].y
        midpoint_x = (left_shoulder_x + right_shoulder_x) / 2
        midpoint_y = (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2
         # Check for mirroring
        if abs(lwrist_y - rwrist_y) < 0.05 and abs(lwrist_x - midpoint_x) - abs(midpoint_x - rwrist_x) < 0.05:
            cv2.putText(image, "MIRRORING ALERT" , (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)

    
    
    cv2.imshow('MediaPipe Pose', image)
    out.write(image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()