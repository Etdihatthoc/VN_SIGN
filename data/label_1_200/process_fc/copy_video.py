import shutil
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv("/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/Son_Tuoi_Hien/video/labels.csv")
    for idx,row in data.iterrows():
        name = row['name']
        shutil.copyfile(f"/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/video/{name}",f"/mnt/disk1/anhnct/HAR/Hand-Sign-Recognition/data/VN_SIGN/Son_Tuoi_Hien/video/{name}")