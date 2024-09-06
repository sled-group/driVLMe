
import os
import csv
from moviepy.editor import VideoFileClip
import moviepy
import multiprocessing
from tqdm.contrib.concurrent import process_map 
from tqdm import tqdm

from drivlme.eval.model_utils import initialize_model, load_video

def parse_video(input_):
    
    video_path,data_points,f_id = input_
    video = VideoFileClip(video_path)
    video = moviepy.video.fx.all.resize(video,(320,180))
    # print(original_size)
    for i,data_point in enumerate(data_points):
        # print(data_point)
        
        try:
            vr = load_video(os.path.join("bdd_100k_preprocessed_test",f"{f_id}_{i}.mp4"))
        except:
            sliced_video = video.subclip(data_point[0], data_point[1])
            final_video = sliced_video.set_fps(5)
            final_video.write_videofile(os.path.join("bdd_100k_preprocessed_test",f"{f_id}_{i}.mp4"), codec='libx264', verbose=False, logger=None)
    return 0

def main():
    root = "bdd100k/videos"
    splits = ["test","train","val"]
    csv_file = "BDD-X-Dataset/BDD-X-Annotations_v1.csv"
    count=0
    data={}
    for split in splits:
        file_path  = os.path.join("BDD-X-Dataset",split+".txt")
        
        with open(file_path, 'r') as file:
            data[split] = [line.strip().split('_')[-1] for line in file.readlines()]
    rows=[]
    with open(csv_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)
    videos=[]
    for row in rows[1:]:
            if row[0]=="":
                continue
            f_name = row[0].split("/")[-1]
            f_id = f_name.split(".")[0]
            start_col = 1
            data_points = []
            while (start_col<len(row)) and(row[start_col]!=""):
                try:
                    data_points.append([int(row[start_col]),int(row[start_col+1]),row[start_col+2],row[start_col+3]])
                    start_col+=4
                except :
                    start_col+=4
                    
            
            if os.path.exists(os.path.join(root,f_name)):
                if f_id not in data["test"]:
                    continue
                video_path = os.path.join(root,f_name)
                for data_point in data_points:
                    
                    count+=1
                videos.append((video_path,data_points,f_id))
    print(count)
    num_processes = multiprocessing.cpu_count()
    print(num_processes)
    results=process_map(parse_video, videos, max_workers=num_processes)
                
        
            
    print(count)
if __name__ == "__main__":
    main()