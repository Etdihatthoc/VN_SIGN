import cv2
import os
from secrets import token_hex
import threading
import pickle
def get_frames(video_name = None,input_dir = "data/AUTSL/data/mp4", output_dir="data/action_status",folder = 'train'):
    """
    Hàm lấy 1 frame đầu, 2 frame giữa, 1 frame cuối từ video và lưu thành ảnh

    Args:
        video_path: Đường dẫn đến video
        output_dir: Thư mục lưu trữ ảnh

    Returns:
        None
    """
    
    
    data_lables = []

    # Đọc video
    cap = cv2.VideoCapture(os.path.join(input_dir,folder,video_name))
    
    video_name = video_name.replace(".mp4","")

    # Lấy số lượng frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tính vị trí các frame cần lấy
    frame_positions = [
        0, # Frame đầu
        frame_count // 2 - 2, # Frame giữa 1
        frame_count // 2 + 2, # Frame giữa 2
        frame_count - 1, # Frame cuối
    ]

    # Duyệt qua các vị trí frame và lưu ảnh
    for i, frame_position in enumerate(frame_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()

        if ret:
            
            file_name = token_hex(10)
            # Ghi ảnh
            cv2.imwrite(os.path.join(output_dir,folder, f"{video_name}_{file_name}.png"), frame)
            if frame_position == 0 or frame_position == frame_count - 1:
                data_lables.append([0,f"{video_name}_{file_name}.png"])
            else:
                data_lables.append([1,f"{video_name}_{file_name}.png"])
        else:
            print(f"Lỗi khi đọc frame {frame_position}")

    # Giải phóng bộ nhớ
    cap.release()
    
    return data_lables


def gen_data(input_dir = "data/AUTSL/data/mp4", output_dir="data/action_status",folder = None):
    assert folder is not None
    os.makedirs(os.path.join(output_dir,folder),exist_ok=True)
    print("Start gen",f" {folder}")
    files = os.listdir(f"{input_dir}/{folder}")
    labels = []
    for file in files:
        if 'depth' not in file:
            lb = get_frames(video_name = file,folder = folder)
            labels.extend(lb)
    
    # Open the file in binary write mode ('wb')
    with open(f"{output_dir}/{folder}.pkl", "wb") as file:
        # Use pickle.dump() to serialize the data and write it to the file
        pickle.dump(labels, file)
    
    
    print("Finish gen",f" {folder}")



if __name__ == "__main__":
    for folder in ['train','val','test']:
        thread = threading.Thread(
            target=gen_data, args=("data/AUTSL/data/mp4","data/action_status",folder))
        thread.start()

    

