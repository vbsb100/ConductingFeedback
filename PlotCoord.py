# Create a Python script that reads in the coordinates from a pickle file and plots them.
# Use the find_peaks function to extract the peaks, plot them. Experiment with the parameters 
# until you get it to work on all 2-4 videos. Plot the coordinates for the 3-4 videos and try to
# figure out how to extract the beats from there. 

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

lwrist_y = []
lwrist_x = []
initial_midpoint_x = 0
midpoint_wristx = 0
frame_count = []
print("What is the pattern for the x and y coordinates?")
name_xy = input()
print("Which option?")
choice_xy = input()
with open('./pickle/lwristx_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
    lwrist_x = pickle.load(f)
with open('./pickle/lwristy_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
    lwrist_y = pickle.load(f)
with open('./pickle/rwristx_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
    rwrist_x = pickle.load(f)
with open('./pickle/rwristy_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
    rwrist_y = pickle.load(f)
with open('./pickle/midpointx_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
    midpoint_x = pickle.load(f)
with open('./pickle/midpointz_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
    midpoint_z = pickle.load(f)
# with open('./pickle/nosex_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
#     nose_x = pickle.load(f)
# with open('./pickle/nosey_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
#     nose_y = pickle.load(f)
# with open('./pickle/lirisx_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
#     liris_x = pickle.load(f)
# with open('./pickle/lirisy_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
#     liris_y = pickle.load(f)
# with open('./pickle/ririsx_'+ name_xy +'(' + choice_xy +').pickle', 'rb') as f:
#     riris_x = pickle.load(f)
# with open('./pickle/ririsy_'+ name_xy +'(' + choice_xy + ').pickle', 'rb') as f:
#     riris_y = pickle.load(f)

for i in range(len(lwrist_x)):
    frame_count.append(i)

midpoint_wristx = (np.array(lwrist_x) + np.array(rwrist_x)) / 2
initial_midwristx = [midpoint_wristx[0]] * len(frame_count)
initial_midpoint_x = [midpoint_x[0]] * len(frame_count)
initial_midpoint_z = [midpoint_z[0]] * len(frame_count)

lwrist_y_peaks, _ = find_peaks(lwrist_y, None, None, 10, 0.0005, None, None, 0.5, None)
lwrist_x_peaks, _ = find_peaks(lwrist_x, None, None, 10, 0.0005, None, None, 0.5, None)
lwrist_y_minima = 1 - np.array(lwrist_y)
#lwrist_x = np.array(lwrist_x)
#lwrist_y_minima = np.array(lwrist_y_minima)
#lwrist = np.square(lwrist_x) + np.square(lwrist_y_minima)
# Plot the x and y coordinate of the wrist joint over time with local maxima and minima
fig, ax = plt.subplots(1)
# ax.plot(frame_count, initial_midpoint_x, label='Initial Midpoint X')
# ax.plot(frame_count, midpoint_x, label='Current Midpoint X')
ax.plot(frame_count, lwrist_y, label='Left Wrist Y')
ax.plot(lwrist_y_peaks,[lwrist_y[i] for i in lwrist_y_peaks], 'x', label='Left Wrist Y Maxima')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Coordinate Value')
ax.set_title('Wrist Coordinates over Time')
ax.legend(loc='upper right')
plt.show()