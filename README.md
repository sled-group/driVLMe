# DriVLMe: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences

### [Project Page](https://sled-group.github.io/driVLMe/) | [Paper](https://arxiv.org/abs/2406.03008) | [Video](https://youtu.be/Ep5fYLGkmsg)

Yidong Huang, Jacob Sansom, Ziqiao Ma, Felix Gervits, Joyce Chai  
University of Michigan, ARL  
IROS 2024

![Method](/method.jpg)

## Setup
The code is adopted from [video-chatgpt](https://github.com/mbzuai-oryx/Video-ChatGPT). We recommend setting up a conda environment for the project:
```shell
conda create --name=drivlme python=3.10
conda activate drivlme

git clone git@github.com:sled-group/driVLMe.git
cd driVLMe
pip install -r requirements.txt
pip install -e .
```

## Prepare Llava weights
Please follow the following instructions to get LLaVA weights.

- Get the original LLaMA weights in the Hugging Face format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
- Use the following scripts to get LLaVA weights by applying our delta.
```shell
python scripts/apply_delta.py \ 
        --base-model-path <path to LLaMA 7B weights> \
        --target-model-path LLaVA-Lightning-7B-v1-1 \
        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
```

The above command will download the LLaVA-Lightening-7B-v1-1 delta from Hugging Face, apply it to the provided LLaMA 
weights and save the LLaVA-Lightening-7B-v1-1 weights in the current directory.

Alternatively you can download the ready LLaVA-Lightening-7B weights from [mmaaz60/LLaVA-Lightening-7B-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1).


## Prepare Dataset
You can get the data at [this dowanload link](https://www.dropbox.com/scl/fo/f429if26mveud6zcek54y/AEjkxF_DZ-DO87xJiOVkQTE?rlkey=shwm81sebtftttkflx8iqghxt&st=eux3itpx&dl=0) and untar all the file under folder "videos".

## Pretrain on bddx dataset

Train on 4 A40 GPUs using the command
```shell
torchrun --nproc_per_node=4 --master_port 29001 drivlme/train/train_xformers.py \
          --model_name_or_path  <Path to Llava> \
          --version v1 \
          --data_path datasets/bddx_pretrain.json \
          --video_folder videos/bdd100k_feats \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./DriVLMe_model_weights/bddx_pretrain_ckpt \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 1000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
```


## Finetune on SDN dataset


Train on 4 A40 GPUs with deepspeed zero2 using the command
```shell
deepspeed --master_port=29001 drivlme/train/train_xformers.py \
          --deepspeed ./scripts/zero2.json \
          --model_name_or_path  <Path to Llava> \
          --pretrain_mm_mlp_adapter ./DriVLMe_model_weights/bddx_pretrain_ckpt/mm_projector.bin \
          --version v1 \
          --lora_enable True \
          --lora_r 128 \
          --lora_alpha 256 \
          --data_path datasets/DriVLMe_sft_data.json \
          --video_folder videos/SDN_train_feats \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./model_path/DriVLMe \
          --num_train_epochs 3 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 500 \
          --save_total_limit 3 \
          --learning_rate 5e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --lazy_preprocess True
```

## Evaluation
You can also download the pretrained checkpoints from [this link](https://www.dropbox.com/scl/fo/neqjdlhohygoa0wrv4uuy/AAjarkE6WY6sKt4LoAfyZ3c?rlkey=e0yvw6g1j8qqdp63vhgi0722d&st=tp2w6h3f&dl=0)

To run the open-loop evaluation, we can use the command 
```shell
python drivlme/single_video_inference_SDN.py --model-name  /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/LLaVA/LLaVA-7B-Lightening-v1-1/ --projection_path ./DriVLMe_model_weights/bddx_pretrain_ckpt/mm_projector.bin --lora_path  ./DriVLMe_model_weights/DriVLMe/ --json_path datasets/SDN_test_actions.json --video_root videos/SDN_test_videos/ --out_path SDN_test_actions.json

python evaluation/physical_action_acc.py
```
for NfD task and 
```shell
python drivlme/single_video_inference_SDN.py --model-name  /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/LLaVA/LLaVA-7B-Lightening-v1-1/ --projection_path ./DriVLMe_model_weights/bddx_pretrain_ckpt/mm_projector.bin --lora_path  ./DriVLMe_model_weights/DriVLMe/ --json_path datasets/SDN_test_conversations.json --video_root videos/SDN_test_videos/ --out_path SDN_test_conversations.json

python evaluation/diag_action_acc.py
```
for RfN task.


## Citation
```bibtex
@misc{huang2024drivlmeenhancingllmbasedautonomous,
      title={DriVLMe: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences}, 
      author={Yidong Huang and Jacob Sansom and Ziqiao Ma and Felix Gervits and Joyce Chai},
      year={2024},
      eprint={2406.03008},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.03008}, 
}
```

## Acknowledgement

We thank the awesome research works [Video-Chatgpt](https://github.com/mbzuai-oryx/Video-ChatGPT), [DriveGPT4](https://tonyxuqaq.github.io/projects/DriveGPT4/), [DriveMLM](https://arxiv.org/abs/2312.09245)



