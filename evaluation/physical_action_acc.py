
import json
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_f1_score(gt_raw, pred):
    # Initialize counts for each class
    classes = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Extract ground truth and predictions
    y_true = gt_raw
    y_pred = pred
    
    # Calculate precision, recall, and F1 score for each class
    f1_scores = {}
    for cls in classes:
        precision = precision_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0)
        f1_scores[cls] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return f1_scores




with open("datasets/SDN_test_actions.json","r")as f:
    gt_raw=json.load(f)
with open("out/SDN_test_actions.json","r")as f:
    pred_raw=json.load(f)
gts=[]
preds=[]
seen_plan_count,unseen_plan_count=0,0
count_seen,count_unseen = 0,0
total_seen,total_unseen = 0,0
slot_count_seen,slot_count_unseen = 0,0
slot_total_seen,slot_total_unseen = 0,0
seen_pred, seen_raw = [], []
unseen_pred, unseen_raw = [], []
for i,pred in enumerate(pred_raw):
    # gts.append(gt_raw[i][ "conversations"][1]["value"])
    # gt_answers = gt_raw[i][ "conversations"][1]["value"].split("\n")
    # gt_answers = [gt_answers[0].split(" ")[2],gt_answers[1].split(" ")[3]]
    # pred_answers = pred.split("\n")
    # pred_answers = [gt_answers[0].split(" ")[2],gt_answers[1].split(" ")[3]]
    # if 
    if gt_raw[i]["unseen"]:
        total_unseen+=1
        if len(gt_raw[i][ "conversations"])>6:
            if gt_raw[i][ "conversations"][-3]["value"][0] == pred[-2][0]:
                count_unseen+=1
            unseen_pred.append(pred[-2][0])
            unseen_raw.append(gt_raw[i][ "conversations"][-3]["value"][0])
            if gt_raw[i][ "conversations"][-1]["value"][0] == pred[-1][0]:
                slot_count_unseen+=1
            slot_total_unseen+=1
        else:
            unseen_pred.append(pred[-1][0])
            unseen_raw.append(gt_raw[i][ "conversations"][-1]["value"][0])
            if gt_raw[i][ "conversations"][-1]["value"][0] == pred[-1][0]:
                count_unseen+=1
            if gt_raw[i][ "conversations"][-3]["value"] == pred[-2]:
                unseen_plan_count+=1
    else:
        total_seen+=1
        if len(gt_raw[i][ "conversations"])>6:
            if gt_raw[i][ "conversations"][-3]["value"][0] == pred[-2][0]:
                count_seen+=1
            seen_pred.append(pred[-2][0])
            seen_raw.append(gt_raw[i][ "conversations"][-3]["value"][0])
            if gt_raw[i][ "conversations"][-1]["value"][0] == pred[-1][0]:
                slot_count_seen+=1
            slot_total_seen+=1
        else:
            seen_pred.append(pred[-1][0])
            seen_raw.append(gt_raw[i][ "conversations"][-1]["value"][0])
            if gt_raw[i][ "conversations"][-1]["value"][0] == pred[-1][0]:
                count_seen+=1
            if gt_raw[i][ "conversations"][-3]["value"] == pred[-2]:
                seen_plan_count+=1
print("unseen:")
print(count_unseen/total_unseen)
print(slot_total_unseen/total_unseen)
print("seen:")

print(count_seen/total_seen)
print(slot_total_seen/total_seen)
# Example usage
    