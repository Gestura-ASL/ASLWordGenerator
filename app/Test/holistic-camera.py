"""
This is the media-pipe implementation
Refer to "https://ai.google.dev/edge/mediapipe/solutions/guide"

##  Real-Time Holistic Landmark Detection

This script implements a real-time computer vision application using **OpenCV** and
 **MediaPipe Holistic** to simultaneously detect and track key landmarks across the face and body (pose)
 using a standard webcam.

### Core Functionality

1.  **Webcam Capture:** Initializes the default camera to stream video frames continuously.
2.  **Holistic Model:** Utilizes MediaPipe's state-of-the-art Holistic model, which integrates multiple
 detection components (Face Mesh, Pose) into a single pipeline.
3.  **Real-Time Processing:** Converts BGR frames to RGB for MediaPipe processing, optimizing performance
 by using a non-writeable flag.
4.  **Landmark Extraction:** Processes each frame to extract coordinates for the body pose skeleton and
the facial contour mesh.
5.  **Visualization:** Draws the extracted landmarks and connections directly onto the original image frame.
6.  **Selfie-View Display:** Displays the annotated frame, flipped horizontally for a natural 'selfie' perspective.

### Key Configuration Parameters

The Holistic model is initialized with the following confidence thresholds:
* `min_detection_confidence=0.5`: The minimum required confidence
for the model to *initially detect* a subject in the frame.
* `min_tracking_confidence=0.5`: The minimum required confidence for
the model to *maintain tracking* of the subject across subsequent frames.

###  Drawing Details

The following specific landmarks are drawn onto the output image:

* **Face:** Drawn using `mp_holistic.FACEMESH_CONTOURS`
to display the outline of the face mesh.
* **Pose:** Drawn using `mp_holistic.POSE_CONNECTIONS` to display the skeletal structure (torso, arms, legs).

### Exit Condition
The application loop runs continuously until the user presses
the **ESC** key (ASCII 27). Upon exiting,
the script properly releases the webcam resource.

"""


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    print(results)

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

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()