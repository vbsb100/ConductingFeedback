# Create a Python script that runs Mediapipe on a video, collects 
# left wrist coordinates in an array (you will probably want both x and y coordinates), 
# and saves the arrays in pickle files. Run it to create these pickle files for all your 
# videos (make sure you name them similarly to the videos). 
import cv2
import numpy as np
import mediapipe as mp
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
video_name = input("\nWhat is the name of your video? ")
pickle_patt = input("\nName the pattern with an underscore in between (Ex. 2_4): ")
choice_xy = input("Input version: ")
video = "./videos/" + video_name + ".MOV"
print("\n" + video)
# For video input
cap = cv2.VideoCapture(video)

# Arrays that hold both frames and wrist coordinates
lwrist_x = []
lwrist_y = []
rwrist_x = []
rwrist_y = []
left_shoulder_x = []
right_shoulder_x = []
left_shoulder_z = []
right_shoulder_z = []
midpoint_x = []
midpoint_z =[]

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  frame_num = 0
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("\nVideo Loaded.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # Resize the image while preserving aspect ratio
    scale_percent = 80 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
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
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)

    # Store the x and y coordinate of the wrist into their designated array
    if results.pose_landmarks:
      lwrist_x.append(results.pose_landmarks.landmark[16].x)
      lwrist_y.append(results.pose_landmarks.landmark[16].y)
      rwrist_x.append(results.pose_landmarks.landmark[15].x)
      rwrist_y.append(results.pose_landmarks.landmark[15].y)

      left_shoulder_x.append(results.pose_landmarks.landmark[11].x)
      right_shoulder_x.append(results.pose_landmarks.landmark[12].x)
      left_shoulder_z.append(results.pose_landmarks.landmark[11].z)
      right_shoulder_z.append(results.pose_landmarks.landmark[12].z)
      midpoint_x = np.array(left_shoulder_x) + np.array(right_shoulder_x)
      midpoint_x /= 2

      midpoint_z = np.array(left_shoulder_z) + np.array(right_shoulder_z)
      midpoint_z /= 2

      
      # Increments frame after a frame is processed
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
with open("./pickle/lwristx_" + pickle_patt + "(" + choice_xy + ").pickle", 'wb') as f:
    pickle.dump(lwrist_x, f)
with open("./pickle/lwristy_" + pickle_patt + "(" + choice_xy + ").pickle", 'wb') as f:
    pickle.dump(lwrist_y, f)  
with open("./pickle/rwristx_" + pickle_patt + "(" + choice_xy + ").pickle", 'wb') as f:
    pickle.dump(rwrist_x, f)
with open("./pickle/rwristy_" + pickle_patt + "(" + choice_xy + ").pickle", 'wb') as f:
    pickle.dump(rwrist_y, f)  
with open("./pickle/midpointx_" + pickle_patt + "(" + choice_xy + ").pickle", 'wb') as f:
    pickle.dump(midpoint_x, f)
with open("./pickle/midpointz_" + pickle_patt + "(" + choice_xy + ").pickle", 'wb') as f:
    pickle.dump(midpoint_z, f)    
  

cap.release()
cv2.destroyAllWindows()