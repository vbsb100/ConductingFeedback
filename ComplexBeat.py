import cv2
import numpy as np
import mediapipe as mp
import pickle
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


lwrist_y = []
lwrist_x = []
frame_count = []
frame_rate = 29.98
prev_beat_frame = None
prev_bpm = -1
beat_num = 1
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

def calculate_slope(lwrist_x, lwrist_y, distance):
    y = np.array(lwrist_y)
    x = np.array(lwrist_x)
    y1 = np.pad(y, (distance, 0), 'constant', constant_values=(0))
    y2 = np.pad(y, (0, distance), 'constant', constant_values=(0))
    x1 = np.pad(x, (distance, 0), 'constant', constant_values=(0))
    x2 = np.pad(x, (0, distance), 'constant', constant_values=(0))
    diff_y = y2 - y1
    diff_x = x2 - x1
    slope = diff_y /diff_x
    return slope

def plot_graphs(slope, lwrist_x, lwrist_y, slope_peaks):
    fig, axs = plt.subplots(3, figsize=(10, 15))

    axs[0].plot(slope, "g")
    axs[0].scatter(slope_peaks, slope[slope_peaks], color = "red", marker = "o")
    axs[0].set_title("Slope")

    
    axs[1].plot(lwrist_y, lwrist_x)
    # axs[1].plot(lwrist_y, "r")
    # axs[1].set_title("Left Wrist Y")

    # axs[2].plot(lwrist_x, "b")
    # axs[2].set_title("Left Wrist X")

    plt.show()


slope = calculate_slope(lwrist_x, lwrist_y, 5)
slope_peaks, _ = find_peaks(slope, None, 0.1, 10, None, None, None, None, None)
plot_graphs(slope, lwrist_x, lwrist_y, slope_peaks)

print(slope[280:300])
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
width2 = int(width * 2)  # Adjust the additional width percentage as needed
out = cv2.VideoWriter('./output/' + name_vid + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width2, height))
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
  first_midpointx = None
  first_midpointz = None
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
        left_shoulder_z = results.pose_landmarks.landmark[11].z
        right_shoulder_z = results.pose_landmarks.landmark[12].z

        lwrist_x = results.pose_landmarks.landmark[16].x
        lwrist_y = results.pose_landmarks.landmark[16].y
        rwrist_x = results.pose_landmarks.landmark[15].x
        rwrist_y = results.pose_landmarks.landmark[15].y
        midpoint_x = (left_shoulder_x + right_shoulder_x) / 2
        midpoint_z = (left_shoulder_z + right_shoulder_z) / 2
        
        if first_midpointx is None:
            first_midpointx = midpoint_x
            first_midpointz = midpoint_z

        threshold = first_midpointx * 0.08
        
    
        if abs(lwrist_y - rwrist_y) < 0.05 and abs(lwrist_x - midpoint_x) - abs(midpoint_x - rwrist_x) < 0.05:
            cv2.putText(image, "MIRRORING ALERT" , (width + 20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        if abs(midpoint_x - first_midpointx) > threshold or abs(midpoint_z - first_midpointz) > threshold:
            cv2.putText(image, "SWAYING ALERT" , (width + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        
        if prev_beat_frame is not None:
            if frame_count.index(frame_num) in slope_peaks:
                num_frames_between_beats = frame_count[frame_count.index(frame_num)] - prev_beat_frame
                num_seconds_between_beats = num_frames_between_beats / frame_rate
                bpm = 60 / num_seconds_between_beats
                beat_num += 1
            else:
                bpm = prev_bpm # Use the previous BPM
        else:
            if frame_count.index(frame_num) in slope_peaks:
                prev_beat_frame = frame_count[frame_count.index(frame_num)]
    # Display the BPM on the video
    if bpm is not None and bpm > -1:
        cv2.putText(image, "BPM: " + str(bpm), (width + 20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
    cv2.putText(image, "Frame Number: " + str(frame_num), (width + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
    
    cv2.imshow('MediaPipe Pose', image)
    out.write(image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    # Update previous beat frame and BPM
    if frame_count.index(frame_num) in slope_peaks:
        prev_beat_frame = frame_count[frame_count.index(frame_num)]
        prev_bpm = bpm # Save the current BPM as the previous BPM
        bpm = None # Reset the current BPM to None
# Update frame count
    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()