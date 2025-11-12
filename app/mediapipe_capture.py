# mediapipe_capture.py
import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def create_frame_landmark_df(results, frame, xyz):
    xyz_skel = (
        xyz[['type', 'landmark_index']]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    face, pose, left_hand, right_hand = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    lefthand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    righthand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, lefthand, righthand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)

    return landmarks


def capture_and_save(xyz_path: str, out_path: str = "output/output.parquet"):
    xyz = pd.read_parquet(xyz_path)
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

            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()

    if all_landmarks:
        pd.concat(all_landmarks).reset_index(drop=True).to_parquet(out_path)
        print(f"Saved landmarks to {out_path}")
        return out_path
    else:
        print("No landmarks captured.")
        return None
