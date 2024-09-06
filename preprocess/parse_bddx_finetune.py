
import os
import csv
from moviepy.editor import VideoFileClip
import moviepy
import multiprocessing
from tqdm.contrib.concurrent import process_map 
from tqdm import tqdm
import json
from random import randrange

from functools import partial
from drivlme.eval.model_utils import initialize_model, load_video

from multiprocessing import Pool


def process_row(root, info_root, data, row):
    if row[0] == "":
        return None
    f_name = row[0].split("/")[-1]
    f_id = f_name.split(".")[0]
    start_col = 1
    data_points = []
    while start_col < len(row) and row[start_col] != "":
        try:
            data_points.append([int(row[start_col]), int(row[start_col+1]), row[start_col+2], row[start_col+3]])
            start_col += 4
        except:
            start_col += 4
                    
    if os.path.exists(os.path.join(root, f_name)) and os.path.exists(os.path.join(info_root, f_id + ".json")):
        if f_id not in data["train"]:
            return None
        video_path = os.path.join(root, f_name)
        info_path = os.path.join(info_root, f_id + ".json")
        with open(info_path, "r") as f:
            info = json.load(f)
        
        return (video_path, info, data_points, f_id)
    return None

def main():
    root = "bdd100k/videos"
    info_root = "bdd100k/info"
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
    with Pool(processes=9) as pool:
        func = partial(process_row, root, info_root, data)
        results = list(tqdm(pool.imap(func, rows[1:]), total=len(rows) - 1))
        # results = list(tqdm(pool.imap(func, rows[1:500]), total=499))

    videos = [result for result in results if result is not None]

    # for row in tqdm(rows[1:]):
    #         if row[0]=="":
    #             continue
    #         f_name = row[0].split("/")[-1]
    #         f_id = f_name.split(".")[0]
    #         start_col = 1
    #         data_points = []
    #         while (start_col<len(row)) and(row[start_col]!=""):
    #             try:
    #                 data_points.append([int(row[start_col]),int(row[start_col+1]),row[start_col+2],row[start_col+3]])
    #                 start_col+=4
    #             except :
    #                 start_col+=4
                    
    #         if os.path.exists(os.path.join(root,f_name))and os.path.exists(os.path.join(info_root,f_id+".json")):
    #             if f_id not in data["train"]:
    #                 continue
    #             video_path = os.path.join(root,f_name)
    #             info_path = os.path.join(info_root,f_id+".json")
    #             with open(info_path,"r") as f:
    #                 info = json.load(f)
    #             for data_point in data_points:
                    
    #                 count+=1
    #             videos.append((video_path,info,data_points,f_id))
    output=[]
    for video in tqdm(videos):
        
        video_path,info,data_points,f_id = video
        for i,data_point in enumerate(data_points):
            hist_loc={"speed":[],"course":[]}
            target_speed=None
            target_course=None
            last_course=None
            for loc in info["locations"]:
                if(loc["timestamp"]<data_point[1]*1000+info["startTime"]):
                    hist_loc["speed"].append(loc["speed"])
                    hist_loc["course"].append(loc["course"])
                if (loc["timestamp"]>=data_point[1]*1000+info["startTime"]) and (target_speed is None):
                    target_speed=loc["speed"]
                if (loc["timestamp"]>=data_point[1]*1000+info["startTime"]) and (target_course is None) and (last_course is not None):
                    target_course=loc["course"]-last_course
                last_course=loc["course"]
            # Convert each speed value to a string, rounded to 2 decimal places
            speed_str = 'Speed(m/s): ' + ', '.join(f'{s:.2f}' for s in hist_loc['speed'])
            turning_angle_str = 'Turning angle(degree): ' + ', '.join(f'{angle:.2f}' for angle in hist_loc['course'])
            if len(hist_loc["speed"])<=1:
                continue
            if target_speed==None:
                target_speed=hist_loc["speed"][-1]
                hist_loc["speed"]=hist_loc["speed"][:-1]
            hist_speed = hist_loc["speed"][-1]
            
            if target_course==None:
                if len(hist_loc["course"])==0:
                    continue
                target_course=hist_loc["course"][-1]-hist_loc["course"][-2]
                hist_loc["course"]=hist_loc["course"][:-1]
                

            if not os.path.exists(os.path.join("bdd100k_feats",f"{f_id}_{i}.pkl")):
                continue
            # try:
            #     vr = load_video(os.path.join("bdd_100k_preprocessed_test",f"{f_id}_{i}.pkl"))
            # except:
            #     print(os.path.join("bdd_100k_preprocessed_test",f"{f_id}_{i}.pkl"))
            #     continue
            
            output_content = {'id': f"{f_id}_{i}", 'video': f"{f_id}_{i}.pkl", 'conversations': []}
            output_content['conversations'].append({'from': 'human', 'value': f"<video>\nGiven this video, the current speed of the vehicle is {hist_speed:.2f} m/s.  What is the current action of this vehicle?\n"})
            output_content['conversations'].append({'from': 'gpt', 'value': data_point[2]})
            output_content['conversations'].append({'from': 'human', 'value': f"Why does the vehicle bahave in this way?",})
            output_content['conversations'].append({'from': 'gpt', 'value': data_point[3] })
            output_content['conversations'].append({'from': 'human', 'value': f"Now predict the next speed of the car in m/s:",})
            output_content['conversations'].append({'from': 'gpt', 'value': f"{target_speed:.2f}" })
            output_content['conversations'].append({'from': 'human', 'value': f"Now predict the next turning angle of the car in degree:",})
            output_content['conversations'].append({'from': 'gpt', 'value': f"{target_course:.2f}" })
            output.append(output_content)
            
            
    
    
    print(len(output))
    with open("bddx_annote_train.json","w") as f:
        json.dump(output,f)
if __name__ == "__main__":
    main()