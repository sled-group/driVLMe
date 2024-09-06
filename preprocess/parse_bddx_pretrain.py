
import os
import csv
from moviepy.editor import VideoFileClip
import moviepy
import multiprocessing
from tqdm.contrib.concurrent import process_map 
from tqdm import tqdm
import json
from random import randrange

from video_chatgpt.eval.model_utils import initialize_model, load_video



def main():
    root = "videos/bdd100k_feats"
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
            # print(f_name)    
            f_name = f_name.split('.')[0]+"_0.pkl"
            if os.path.exists(os.path.join(root,f_name)):
                if f_id not in data["train"]:
                    continue
                video_path = os.path.join(root,f_name)
                for data_point in data_points:
                    
                    count+=1
                videos.append((video_path,data_points,f_id))
    print(count)
    output=[]
    for video in tqdm(videos):
        
        video_path,data_points,f_id = video
        for i,data_point in (enumerate((data_points))):
            if not os.path.exists(os.path.join("videos/bdd100k_feats",f"{f_id}_{i}.pkl")):
                continue
            # try:
            #     vr = load_video(os.path.join("bdd_100k_preprocessed_test",f"{f_id}_{i}.mp4"))
            # except:
            #     print(os.path.join("bdd_100k_preprocessed_test",f"{f_id}_{i}.mp4"))
            #     continue
            
            output_content = {'id': f"{f_id}_{i}", 'video': f"{f_id}_{i}.pkl", 'conversations': []}
            rnd=randrange(2)
            if rnd ==0:
                output_content['conversations'].append({'from': 'human', 'value': f"Given this video, please tell me what this car is doing\n<video>"})
                output_content['conversations'].append({'from': 'gpt', 'value': data_point[2]+' '+data_point[3] })
            else:
                output_content['conversations'].append({'from': 'human', 'value': f"<video>\nGiven this video, please tell me what this car is doing"})
                output_content['conversations'].append({'from': 'gpt', 'value': data_point[2]+' '+data_point[3] })
            output.append(output_content)
            
            output_content = {'id': f"{f_id}_{i}", 'video': f"{f_id}_{i}.pkl", 'conversations': []}
            rnd=randrange(2)
            if rnd ==0:
                output_content['conversations'].append({'from': 'human', 'value': f"Given this video, please tell me why: {data_point[2]} \n<video>"})
                output_content['conversations'].append({'from': 'gpt', 'value': data_point[2]+' '+data_point[3] })
            else:
                output_content['conversations'].append({'from': 'human', 'value': f"<video>\nGiven this video, please tell me why: {data_point[2]}"})
                output_content['conversations'].append({'from': 'gpt', 'value': data_point[2]+' '+data_point[3] })
            output.append(output_content)
            
            output_content = {'id': f"{f_id}_{i}", 'video': f"{f_id}_{i}.pkl", 'conversations': []}
            rnd=randrange(2)
            if rnd ==0:
                output_content['conversations'].append({'from': 'human', 'value': f"In this video, what will the car do given the reason: {data_point[3]} \n<video>"})
                output_content['conversations'].append({'from': 'gpt', 'value': data_point[2]+' '+data_point[3] })
            else:
                output_content['conversations'].append({'from': 'human', 'value': f"<video>\nIn this video, what will the car do given the reason: {data_point[3]} "})
                output_content['conversations'].append({'from': 'gpt', 'value': data_point[2]+' '+data_point[3] })
            output.append(output_content)
            
    
    
    print(len(output))
    with open("datasets/bddx_pretrain.json","w") as f:
        json.dump(output,f)
if __name__ == "__main__":
    main()