import cv2
import mediapipe as mp
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic



def create_frame_landmark_df(results, frame, xyz):
    """
Takes MediaPipe results and creates a DataFrame of landmarks
Input: results, frame number, and xyz skeleton
output: DataFrame with (frame, type, landmark_index, x, y, z)


The below function makes the data taken from mediapipe
into the desiered shape that we need

structure:
	type	landmark_index	x	y	z	frame
0	face	0	0.327897	0.606235	-0.027963	0
1	face	1	0.323504	0.561802	-0.043389	0
2	face	2	0.329605	0.577532	-0.025045	0
3	face	3	0.323498	0.526857	-0.026195	0
4	face	4	0.323451	0.549734	-0.045150	0
...	...	...	...	...	...	...
538	right_hand	16	NaN	NaN	NaN	0
539	right_hand	17	NaN	NaN	NaN	0
540	right_hand	18	NaN	NaN	NaN	0
541	right_hand	19	NaN	NaN	NaN	0
542	right_hand	20	NaN	NaN	NaN	0
"""

    xyz_skel = (
        xyz[['type', 'landmark_index']]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    # Initialize empty dataframes
    face, pose, left_hand, right_hand = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---- FACE ----
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # ---- POSE ----
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # ---- LEFT HAND ----
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # ---- RIGHT HAND ----
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # Assign landmark indices and types
    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    lefthand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    righthand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    # Combine and align with xyz skeleton
    landmarks = pd.concat([face, pose, lefthand, righthand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)

    return landmarks


def do_capture_loop(xyz):

    """
    media-pipe implementation - refer holistic-camera.py for more explaination

    """

    # Create a skeleton like Kaggle’s xyz format
   # xyz = pd.concat([
    #    pd.DataFrame({'type': 'face', 'landmark_index': range(468)}),
     #   pd.DataFrame({'type': 'pose', 'landmark_index': range(33)}),
      #  pd.DataFrame({'type': 'left_hand', 'landmark_index': range(21)}),
      #  pd.DataFrame({'type': 'right_hand', 'landmark_index': range(21)}),
    #]).reset_index(drop=True)

    all_landmarks = []


    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame = 0
        while cap.isOpened():
            frame += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            landmarks = create_frame_landmark_df(results, frame, xyz)
            all_landmarks.append(landmarks)

            # Draw landmarks
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Display the frame
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                break



    return all_landmarks



if __name__ == '__main__':

    pq_file = "../../asl-signs/train_landmark_files/16069/10042041.parquet"
    xyz = pd.read_parquet(pq_file)
    landmarks = do_capture_loop(xyz)
    if landmarks:
        pd.concat(landmarks).reset_index(drop=True).to_parquet("output.parquet")
        print("Saved landmarks to output.parquet")
    else:
        print("No landmarks captured.")
