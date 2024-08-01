from mmpose.apis import MMPoseInferencer
import numpy as np
import cv2
import time
import json
import os
import pandas as pd
"""
# LeftHand : [92,111]
# RightHand: [113,132] include 10
# Pose: [0,10] include 10

# After append neck and headtop at first:

# LeftHand : [94,113] include 113
# RightHand: [115,134] include 134
# Pose: [0,10] include 10
# Neck: 17 HeadTop: 18
"""

def read_and_write_video(input_video_path, output_video_path,keypoints = None):
    assert keypoints is not None
    """
    Đọc video từ đường dẫn đầu vào và ghi lại video vào đường dẫn đầu ra.
    
    Args:
    - input_video_path (str): Đường dẫn đến video đầu vào.
    - output_video_path (str): Đường dẫn đến video đầu ra.
    """
    # Mở video đầu vào
    cap = cv2.VideoCapture(input_video_path)
    
    # Lấy thông số của video đầu vào
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tạo video writer để ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    cnt = 0
    keypoints = np.array(keypoints).reshape(-1,135,3)
    # Đọc từng frame từ video đầu vào và ghi vào video đầu ra
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        wholebody = keypoints[cnt].tolist()
        # pose
        for kp in wholebody[0:13]:
            cv2.circle(frame, (int(kp[0]),int(kp[1])), 3, (0, 255, 0), -1)
        # lefthand
        for kp in wholebody[94:114]:
            cv2.circle(frame, (int(kp[0]),int(kp[1])), 3, (0, 255, 0), -1)
        # right hand
        for kp in wholebody[115:135]:
            cv2.circle(frame, (int(kp[0]),int(kp[1])), 3, (0, 255, 0), -1)
        
        out.write(frame)
        cnt+=1
    # Giải phóng tài nguyên
    cap.release()
    out.release()

def gen_pose(base_url,file_name,wholebody_detector,pose_detector):
    video_url = os.path.join(base_url,file_name)
    wholebody_results = wholebody_detector(video_url)
    pose_results = pose_detector(video_url)

    kp_folder = video_url.replace("video",'kp').replace('.mp4',"")
    os.makedirs(kp_folder,exist_ok=True)
    
    for idx,(pose_result,wholebody_result) in enumerate(zip(pose_results,wholebody_results)):
        wholebody = wholebody_result['predictions'][0][0]['keypoints'][:16] + pose_result['predictions'][0][0]['keypoints'][17:19][::-1] +  wholebody_result['predictions'][0][0]['keypoints'][16:]
        prob = wholebody_result['predictions'][0][0]['keypoint_scores'][:16] + pose_result['predictions'][0][0]['keypoint_scores'][17:19][::-1] +  wholebody_result['predictions'][0][0]['keypoint_scores'][16:]
        raw_wholebody = [[value[0],value[1],0] for idx,value in enumerate(wholebody)]
        wholebody_threshold_03 = [[value[0],value[1],0] if prob[idx] > 0.3 else [0,0,0] for idx,value in enumerate(wholebody)]
        wholebody_threshold_02 = [[value[0],value[1],0] if prob[idx] > 0.2 else [0,0,0] for idx,value in enumerate(wholebody)]
        dict_data = {
            "raw_wholebody": raw_wholebody,
            "wholebody_threshold_02": wholebody_threshold_02,
            "wholebody_threshold_03": wholebody_threshold_03,
            "prob": prob
        }
        dest = os.path.join(kp_folder,file_name.replace(".mp4","") + '_{:06d}_'.format(idx) + 'keypoints.json')
        with open(dest, 'w') as f:
            json.dump(dict_data, f)
        
       


if __name__ == "__main__":
    # full_data = pd.read_csv("/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/full_data_1_200.csv")[:4000]
    # full_data = pd.read_csv("/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/full_data_1_200.csv")[4000:8000]
    # full_data = pd.read_csv("/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/full_data_1_200.csv")[8000:12000]
    full_data = pd.read_csv("/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/full_data_1_200.csv")[12000:]
    wholebody_detector = MMPoseInferencer( "td-hm_res152_8xb32-210e_coco-wholebody-384x288")
    pose_detector = MMPoseInferencer( "rtmpose-m_8xb512-700e_body8-halpe26-256x192")
    print(full_data.shape)

    for idx,data in full_data.iterrows():
        gen_pose("/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/video",data['name'],wholebody_detector,pose_detector)

    
   