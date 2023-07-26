# Create a Python script that reads in the coordinates from a pickle file 
# and detects the peaks, and then runs through the video displaying not only
# the skeleton, but also a flashing circle whenever a beat is detected (you 
# can make that circle bigger by the way). Compute the tempo (beats per minute).
import cv2
import numpy as np
import mediapipe as mp
import pickle
from scipy.signal import find_peaks
import numpy as np

lwrist_y = []
lwrist_x = []
frame_count = []
frame_rate = 29.98
prev_beat_frame = None
prev_bpm = -1
bpm = None
frame_num = 0
print("What is the pattern for the x and y coordinates?")
name_xy = input()
print("Which option?")
choice_xy = input()
print("What is the video file name?")
name_vid = input()
with open('./pickle/lwristx_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
    lwrist_x = pickle.load(f)
with open('./pickle/lwristy_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
    lwrist_y = pickle.load(f)

for i in range(len(lwrist_y)):
    frame_count.append(i)

lwrist_y_peaks, _ = find_peaks(lwrist_y, None, None, 10, 0.0005, None, None, 0.5, None)
lwrist_x_peaks, _ = find_peaks(lwrist_x, None, None, 10, 0.0005, None, None, 0.5, None)

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
# Display flashing circle at frames that have a beat
    if results.pose_landmarks:
        if prev_beat_frame is not None:
            if frame_count.index(frame_num) in lwrist_y_peaks:
                #Display BPM average over 10 peaks using a queue, pushing or popping
                num_frames_between_beats = frame_count[frame_count.index(frame_num)] - prev_beat_frame
                num_seconds_between_beats = num_frames_between_beats / frame_rate
                bpm = 60 / num_seconds_between_beats
                for i in range(10):
                        cv2.circle(image, (70, 100), 50, (255, 255, 0), 10)
            else:
                bpm = prev_bpm # Use the previous BPM
        else:
            if frame_count.index(frame_num) in lwrist_y_peaks:
                prev_beat_frame = frame_count[frame_count.index(frame_num)]

    # Display the BPM on the video
    if bpm is not None and bpm > -1:
        cv2.putText(image, "BPM: " + str(bpm), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
    # Still need to add real-time accurate BPM algorithm
    
    cv2.imshow('MediaPipe Pose', image)
    out.write(image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    # Update previous beat frame and BPM
    if frame_count.index(frame_num) in lwrist_y_peaks:
        prev_beat_frame = frame_count[frame_count.index(frame_num)]
        prev_bpm = bpm # Save the current BPM as the previous BPM
        bpm = None # Reset the current BPM to None
# Update frame count
    frame_num += 1
 
cap.release()
out.release()
cv2.destroyAllWindows()