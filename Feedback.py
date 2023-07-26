import cv2
import numpy as np
import mediapipe as mp
import pickle
from scipy.signal import find_peaks
import numpy as np

# Initialize variables
lwrist_y = []
lwrist_x = []
frame_count = []
frame_rate = 29.98
prev_beat_frame = None
prev_bpm = -1
beat_num = 1
bpm = None
frame_num = 0

# User input prompt
print("What is the pattern for the x and y coordinates?")
name_xy = input()
print("Which option?")
choice_xy = input()
print("What is the video file name?")
name_vid = input()

# Load wrist coordinate data from pickle files
with open('./pickle/lwristx_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
    lwrist_x = pickle.load(f)
with open('./pickle/lwristy_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
    lwrist_y = pickle.load(f)

# Get frame count and find peaks from wrist data
for i in range(len(lwrist_y)):
    frame_count.append(i)

lwrist_y_peaks, _ = find_peaks(lwrist_y, None, None, 10, 0.0005, None, None, 0.5, None)
lwrist_x_peaks, _ = find_peaks(lwrist_x, None, None, 10, 0.0005, None, None, 0.5, None)

# MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For video input
cap = cv2.VideoCapture("./videos/" + name_vid + ".MOV")
fps =  cap.get(cv2.CAP_PROP_FPS)

# Resize the image while preserving aspect ratio
scale_percent = 80 # percent of original size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent / 100)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent / 100)
width2 = int(width * 2)  # Adjust the additional width percentage as needed

# Output video
out = cv2.VideoWriter('./output/' + name_vid + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width2, height))


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

  # Initialize first midpoint variables 
  first_midpointx = None
  first_midpointz = None
  nose_midpoint_x = None
  nose_midpoint_y = None

  # Processing each frame
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
 
    
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Adds black area to the right of the image frame
    black_area = np.zeros((height, int(width), 3), dtype=np.uint8)  
    image = np.concatenate((image, black_area), axis=1)

    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    # Store shoulder and wrist data from landmarks
    if results.pose_landmarks:
        left_shoulder_x = results.pose_landmarks.landmark[11].x
        right_shoulder_x = results.pose_landmarks.landmark[12].x
        left_shoulder_z = results.pose_landmarks.landmark[11].z
        right_shoulder_z = results.pose_landmarks.landmark[12].z

        lwrist_x = results.pose_landmarks.landmark[16].x
        lwrist_y = results.pose_landmarks.landmark[16].y
        rwrist_x = results.pose_landmarks.landmark[15].x
        rwrist_y = results.pose_landmarks.landmark[15].y

        

        midpoint_x = (left_shoulder_x + right_shoulder_x) / 2
        midpoint_z = (left_shoulder_z + right_shoulder_z) / 2
        
        
        # Set the threshold for swaying
        if first_midpointx is None:
            first_midpointx = midpoint_x
            first_midpointz = midpoint_z

        threshold = first_midpointx * 0.08
        
        # Check for mirroring alert
        if abs(lwrist_y - rwrist_y) < 0.05 and abs(lwrist_x - midpoint_x) - abs(midpoint_x - rwrist_x) < 0.05:
            cv2.putText(image, "MIRRORING ALERT" , (width + 20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        # Check for swaying alert
        if abs(midpoint_x - first_midpointx) > threshold or abs(midpoint_z - first_midpointz) > threshold:
            cv2.putText(image, "SWAYING ALERT" , (width + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        
        # Check if the current frame is a peak in wrist y
        if prev_beat_frame is not None:
            if frame_count.index(frame_num) in lwrist_y_peaks:
                # Calculate BPM based on the number of frames between beats
                num_frames_between_beats = frame_count[frame_count.index(frame_num)] - prev_beat_frame
                num_seconds_between_beats = num_frames_between_beats / frame_rate
                bpm = 60 / num_seconds_between_beats
                beat_num += 1
            else:
                bpm = prev_bpm # Use the previous BPM
        else:
            if frame_count.index(frame_num) in lwrist_y_peaks:
                # Set previous beat frame to the current frame if peak detected
                prev_beat_frame = frame_count[frame_count.index(frame_num)]

    # Display the BPM and beat number on the video
    if bpm is not None and bpm > -1:
        cv2.putText(image, "BPM: " + str(bpm), (width + 20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
    cv2.putText(image, "BEAT #: " + str(beat_num), (width + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)

    if results.face_landmarks:
        if nose_midpoint_x is None:
            nose_midpoint_x = results.face_landmarks.landmark[4].x
            nose_midpoint_y = results.face_landmarks.landmark[4].y
        nose_threshold_x = nose_midpoint_x * 0.06
        nose_threshold_y = nose_midpoint_y * 0.06

        
        # Check the direction of nose to detect where the user is looking
        if (abs(results.face_landmarks.landmark[4].x - nose_midpoint_x)) <  nose_threshold_x:
            cv2.putText(image,"Looking straight", (width + 20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        elif (abs(results.face_landmarks.landmark[4].x - nose_midpoint_x)) > nose_threshold_x:
            cv2.putText(image,"Looking to the side", (width + 20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        elif (abs(results.face_landmarks.landmark[4].y - nose_midpoint_y)) > nose_threshold_y:
            cv2.putText(image,"Looking down", (width + 20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        

    # Display and write image to an output file    
    cv2.imshow('MediaPipe Holistic', image)
    out.write(image)

    # Loop breaks if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    # Update previous beat frame and BPM
    if frame_count.index(frame_num) in lwrist_y_peaks:
        prev_beat_frame = frame_count[frame_count.index(frame_num)]
        prev_bpm = bpm # Save the current BPM as the previous BPM
        bpm = None # Reset the current BPM to None
# Update frame count
    frame_num += 1

# Release the video capture, output write, and close all windows 
cap.release()
out.release()
cv2.destroyAllWindows()