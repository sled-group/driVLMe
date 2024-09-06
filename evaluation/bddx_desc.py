import csv
import json
import os
import os.path as op
import re
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict, defaultdict
from pprint import pprint
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from ciderD.ciderD import CiderD


def score_all(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider("corpus"), "Cider"),
        # (CiderD(), "CiderD")
    ]
    final_scores = {}
    for scorer,method in tqdm(scorers):
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores 

with open("datasets/bddx_annote_test.json","r")as f:
    gt_raw=json.load(f)
with open("out/bddx_annote_test.json","r")as f:
    pred_raw=json.load(f)
print(len(pred_raw))
print(len(gt_raw))
preds=dict()
gts=dict()
for i in range(len(gt_raw)):
    
# for i,pred in enumerate(pred_raw):
        preds[i]=[pred_raw[i][0]+" " +pred_raw[i][1]]
        gts[i] = [gt_raw[i][ "conversations"][1]["value"]+" " +gt_raw[i][ "conversations"][3]["value"]]
        # preds[i]=[pred_raw[i][1]]
        # gts[i] = [gt_raw[i][ "conversations"][3]["value"]]
print(preds[1450],gts[1450])
scores = score_all(gts, preds)
print(scores)

