
import numpy as np
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm

with open("datasets/bddx_annote_test.json","r")as f:
    gt_raw=json.load(f)
with open("out/bddx_annote_test.json","r")as f:
    pred_raw=json.load(f)
speed_RMSE = []
course_RMSE = []
speed_A01=0
speed_A05=0
speed_A1=0
speed_A5=0
course_A01=0
course_A05=0
course_A1=0
course_A5=0
for i,pred in enumerate(pred_raw):
    pred_speed = float(pred[-2])
    gt_speed = float(gt_raw[i][ "conversations"][5]["value"])
    speed_RMSE.append((pred_speed-gt_speed)**2)
    if abs(pred_speed-gt_speed)<=0.1:
        speed_A01+=1
    if abs(pred_speed-gt_speed)<=0.5:
        speed_A05+=1
    if abs(pred_speed-gt_speed)<=1:
        speed_A1+=1
    if abs(pred_speed-gt_speed)<=5:
        speed_A5+=1
    try:
        pred_course = float(pred[-1])
    except:
        pred_course=0
    gt_course = float(gt_raw[i][ "conversations"][7]["value"])
    course_RMSE.append((pred_course-gt_course)**2)
    if abs(pred_course-gt_course)<=0.1:
        course_A01+=1
    if abs(pred_course-gt_course)<=0.5:
        course_A05+=1
    if abs(pred_course-gt_course)<=1:
        course_A1+=1
    if abs(pred_course-gt_course)<=5:
        course_A5+=1
print("speed_RMSE",np.sqrt(sum(speed_RMSE)/len(speed_RMSE)))
print("speed_A01",speed_A01/len(pred_raw))
print("speed_A05",speed_A05/len(pred_raw))
print("speed_A1",speed_A1/len(pred_raw))
print("speed_A5",speed_A5/len(pred_raw))
print("course_RMSE",np.sqrt(sum(course_RMSE)/len(course_RMSE)))
print("course_A01",course_A01/len(pred_raw))
print("course_A05",course_A05/len(pred_raw))
print("course_A1",course_A1/len(pred_raw))
print("course_A5",course_A5/len(pred_raw))
    