import cv2
import mediapipe as mp
import time  
import threading
import pickle as pk
import pandas as pd
import pyttsx4
import multiprocessing as mtp
import os

from recommendations import check_pose_angle
from landmarks import extract_landmarks
from calc_angles import rangles

stop_flag = False

def init_cam():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FOCUS, 360)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 130)
    cam.set(cv2.CAP_PROP_SHARPNESS, 125)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam

def get_pose_name(index):
    names = {
        0: "cobra",
        1: "downdog",
        2: "goddess",
        3: "mountain",
        4: "plankup",
        5: "tree",
        6: "warrior",
    }
    return str(names[index])

def get_pose_index(name):
    reverse_names = {
        "cobra": 0,
        "downdog": 1,
        "goddess": 2,
        "mountain": 3,
        "plankup": 4,
        "tree": 5,
        "warrior": 6,
    }
    return reverse_names.get(name, -1)

def init_dicts():
    landmarks_points = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_heel": 29, "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    }
    landmarks_points_array = {
        "left_shoulder": [], "right_shoulder": [],
        "left_elbow": [], "right_elbow": [],
        "left_wrist": [], "right_wrist": [],
        "left_hip": [], "right_hip": [],
        "left_knee": [], "right_knee": [],
        "left_ankle": [], "right_ankle": [],
        "left_heel": [], "right_heel": [],
        "left_foot_index": [], "right_foot_index": [],
    }
    col_names = []
    for i in range(len(landmarks_points.keys())):
        name = list(landmarks_points.keys())[i]
        col_names.append(name + "_x")
        col_names.append(name + "_y")
        col_names.append(name + "_z")
        col_names.append(name + "_v")
    cols = col_names.copy()
    return cols, landmarks_points_array

engine = pyttsx4.init()

def tts(tts_q):
    while True:
        objects = tts_q.get()
        if objects is None:
            break
        message = objects[0]
        engine.say(message)
        engine.runAndWait()
    tts_q.task_done()

def cv2_put_text(image, pose_name, hold_time, mae=None):
    cv2.putText(
        image,
        f"Pose: {pose_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        image,
        f"Hold Time: {int(hold_time)}s",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    if mae is not None:
        cv2.putText(
            image,
            f"MAE: {mae:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

def stop_pose_detection():
    global stop_flag
    stop_flag = True
    print("Yoga session stopped.")

def destory(cam, tts_proc, tts_q):
    cv2.destroyAllWindows()
    cam.release()
    tts_q.put(None)
    tts_q.close()
    tts_q.join_thread()
    tts_proc.join()

# ... (imports and initial setup remain unchanged)

def start_pose_detection():
    global stop_flag
    pose_start_time = None
    hold_time = 0
    hold_time_threshold = 3

    cam = init_cam()

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'poses.model')
    model = pk.load(open(model_path, 'rb'))

    cols, landmarks_points_array = init_dicts()
    angles_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'csv_files', 'poses_angles.csv'))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    tts_q = mtp.JoinableQueue()
    tts_proc = mtp.Process(target=tts, args=(tts_q,))
    tts_proc.start()

    tts_last_exec = time.time() + 5
    counter_feedback_time = None

    while not stop_flag:
        result, image = cam.read()
        flipped = cv2.flip(image, 1)
        resized_image = cv2.resize(flipped, (640, 360), interpolation=cv2.INTER_AREA)

        key = cv2.waitKey(1)
        if key == ord("q"):
            stop_pose_detection()
            destory(cam, tts_proc, tts_q)
            break

        if result:
            err, df, landmarks = extract_landmarks(resized_image, mp_pose, cols)

            if err == False:
                prediction = model.predict(df)
                probabilities = model.predict_proba(df)

                mp_drawing.draw_landmarks(flipped, landmarks, mp_pose.POSE_CONNECTIONS)

                if probabilities[0, prediction[0]] > 0.85:
                    pose_name = get_pose_name(prediction[0])
                    pose_index = prediction[0]

                    if pose_start_time is None:
                        pose_start_time = time.time()

                    hold_time = time.time() - pose_start_time
                    angles = rangles(df, landmarks_points_array)
                    mae = None
                    pose_row = None

                    if angles and len(angles) > 0:
                        pose_row = angles_df[angles_df["class"] == pose_index]

                        if not pose_row.empty:
                            ideal_angles = pose_row.drop("class", axis=1).values[0]
                            differences = [abs(a - b) for a, b in zip(angles, ideal_angles)]
                            mae = sum(differences) / len(angles)
                        else:
                            mae = None
                    else:
                        mae = None

                    cv2_put_text(flipped, pose_name, hold_time, mae)

                    if hold_time >= hold_time_threshold:
                        if counter_feedback_time is None or time.time() - counter_feedback_time > 5:
                            tts_q.put([f"Great job! You've held the {pose_name} pose for {int(hold_time)} seconds."])
                            counter_feedback_time = time.time()
                        pose_start_time = None

                    suggestions = check_pose_angle(pose_index, angles, angles_df)

                    if time.time() > tts_last_exec:
                        tts_q.put([suggestions[0]])
                        tts_last_exec = time.time() + 5

                else:
                    cv2_put_text(flipped, "No pose detected", 0)

            cv2.imshow("Frame", flipped)

    stop_pose_detection()
    destory(cam, tts_proc, tts_q)

if __name__ == "__main__":
    start_pose_detection()