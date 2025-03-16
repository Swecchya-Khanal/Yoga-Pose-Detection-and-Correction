import pandas as pd
import numpy as np
import cv2


def extract_landmarks(image, mp_pose, cols):
    pre_list = []
    with mp_pose.Pose(static_image_mode=True, enable_segmentation=True) as pose:
        result = pose.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            
            for landmark in result.pose_landmarks.landmark:
                pre_list.append(landmark)
            predict = True
        except AttributeError:
            return True, pd.DataFrame(), None

    if predict == True:
        gen1116 = np.array([
            [
                pre_list[m].x,
                pre_list[m].y,
                pre_list[m].z,
                pre_list[m].visibility
            ] for m in range(11, 17)
        ]).flatten().tolist()

        gen2333 = np.array([
            [
                pre_list[m].x,
                pre_list[m].y,
                pre_list[m].z,
                pre_list[m].visibility
            ] for m in range(23, 33)
        ]).flatten().tolist()

        gen1116.extend(gen2333)

        all_list = [
            pre_list[0].x,
            pre_list[0].y,
            pre_list[0].z,
            pre_list[0].visibility,
        ]

        all_list.extend(gen1116)
        return False, pd.DataFrame([all_list], columns=cols), result.pose_landmarks

