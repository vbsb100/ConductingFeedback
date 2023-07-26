import cv2
import mediapipe as mp
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

left_iris_x = []
right_iris_x = []
nose_x = []
left_iris_y = []
right_iris_y = []
nose_y = []
print("What is the eyecontact video version?")
opt = input()
# For video input
cap = cv2.VideoCapture("./videos/eyecontact" + opt + ".MOV")
fps =  cap.get(cv2.CAP_PROP_FPS)
# Resize the image while preserving aspect ratio
scale_percent = 80 # percent of original size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent / 100)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent / 100)
out = cv2.VideoWriter('./output/eyecontact' + opt + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
frame = 0
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

  nose_midpoint_x = None 
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
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      #for face_landmarks in results.multi_face_landmarks:
      face_landmarks = results.multi_face_landmarks[0]
      mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style()) 
      mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
      # shape = image.shape
      # cx = int(face_landmarks.landmark[468].x * shape[1])
      # cy = int(face_landmarks.landmark[468].y * shape[0])
      # cz = face_landmarks.landmark[468].z
      
      # Flip the image horizontally for a selfie-view display. Check Week 3 CANVAS
      # Add looking downright or downleft, for clarinets, flute
      # GRAPH ALL OF THESE AND VIEW THE COORDINATES
      if face_landmarks.landmark:
        if nose_midpoint_x is None:
            nose_midpoint_x = face_landmarks.landmark[4].x   
        nose_threshold = nose_midpoint_x * 0.06
        
        # Check the direction of nose to detect where the user is looking
        if (abs(face_landmarks.landmark[4].x - nose_midpoint_x)) <  nose_threshold:
            cv2.putText(image,"Looking straight", (220, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        elif face_landmarks.landmark[4].x > nose_midpoint_x + nose_threshold:
            cv2.putText(image,"Looking to the right", (220, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        elif face_landmarks.landmark[4].x < nose_midpoint_x + nose_threshold:
            cv2.putText(image,"Looking to the left", (220, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
    # cv2.putText(image,"NOSE:" + str(face_landmarks.landmark[4].x) + " y:"+ str(face_landmarks.landmark[4].y) + " z:"+ str(face_landmarks.landmark[4].z), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 4)
    # cv2.putText(image,"Left Iris:" + str(face_landmarks.landmark[468].x) + " y:"+ str(face_landmarks.landmark[468].y) + " z:"+ str(face_landmarks.landmark[468].z), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
    # cv2.putText(image,"Right Iris:" + str(face_landmarks.landmark[473].x) + " y:"+ str(face_landmarks.landmark[473].y) + " z:"+ str(face_landmarks.landmark[473].z), (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
    # cv2.circle(image, (cx, cy), 1, (255, 255, 0), 5)
    cv2.imshow('MediaPipe Face Mesh', image)
    # print("x:" + str(cx) +"y:" + str(cy))
    out.write(image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    with open("./pickle/lirisx_eyecontact(" + opt + ").pickle", 'wb') as f:
        pickle.dump(left_iris_x, f)
    with open("./pickle/lirisy_eyecontact(" + opt + ").pickle", 'wb') as f:
        pickle.dump(left_iris_y, f)
    with open("./pickle/ririsx_eyecontact(" + opt + ").pickle", 'wb') as f:
        pickle.dump(right_iris_x, f)
    with open("./pickle/ririsy_eyecontact(" + opt + ").pickle", 'wb') as f:
        pickle.dump(right_iris_y, f)  
    with open("./pickle/nosex_eyecontact(" + opt + ").pickle", 'wb') as f:
        pickle.dump(nose_x, f)
    with open("./pickle/nosey_eyecontact(" + opt + ").pickle", 'wb') as f:
        pickle.dump(nose_y, f)    

cap.release()
out.release()
cv2.destroyAllWindows()