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

import sys
sys.path.append("/DATA_EDS/zyp/jinbu/drama/")

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from evalcap.cider.pyciderevalcap.ciderD.ciderD import CiderD


def score_all(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(), "Cider"),
        (CiderD(), "CiderD")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores 

with open("pred.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    preds_list = [(row['public_id'], row['caption'].strip('.')) for row in reader]

preds = dict()
for pred in preds_list:
    preds[f'{pred[0]}'] = [pred[1]]

with open("gt.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    gts_list = [(row['public_id'], row['caption_gt']) for row in reader]

gts = dict()
for gt in gts_list:
    gts[f'{gt[0]}'] = [gt[1]]

scores = score_all(gts, preds)
print(scores)

