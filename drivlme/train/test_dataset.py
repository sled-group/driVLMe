import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from drivlme.train.llava_trainer import VideoChatGPTTrainer
from drivlme import video_conversation as conversation_lib
from drivlme.model import *
import torch.distributed as dist
from drivlme.constants import *
import pickle
import os
from tqdm import tqdm
from train import ModelArguments, DataArguments, TrainingArguments, make_supervised_data_module


DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
def test():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    data_args.video_token_len = 10
    data_args.is_multimodal = True

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["train_dataset"]
    collector = data_module["data_collator"]
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        print(collector["video"]["CAM_FRONT"])
        print(collector([item])['video_spatio_temporal_features']["CAM_FRONT"].shape)
test()
    
    