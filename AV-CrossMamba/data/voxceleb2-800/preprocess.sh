#!/bin/bash 

# --- Base Directory Configuration ---
direc=/home/wangcanyang/vox2

data_direc=${direc}/

# --- Simulation Parameters ---
train_samples=20000 # 训练集混合样本数
val_samples=5000    # 验证集混合样本数
test_samples=3000   # 测试集混合样本数

# --- NEW MIXTURE PARAMETERS ---
C=4 # 混合中的扬声器数量 (3人)
mix_db=10 # 随机 dB 比例范围 (-10 dB 到 +10 dB)
min_length=6 # 最小音频长度 (6 秒)

# --- File Naming ---
# 更新文件名以反映 3mix 和 6s 的新配置
mixture_data_list=mixture_data_list_${C}mix_${min_length}s.csv

# --- Fixed Parameters ---
sampling_rate=16000 # 音频采样率 (16kHz)

# --- Path Configuration ---
audio_data_direc=${direc}clean_audio/ # 干净音频保存目录
# 更新输出目录以反映 3mix 和 6s 的新配置
mixture_audio_direc=${direc}_3mix/${C}_mix_${min_length}s/
visual_frame_direc=${direc}face/       # 视觉帧保存目录 (未在此脚本中使用，但保留)
lip_embedding_direc=${direc}lip/      # 唇部嵌入保存目录 (未在此脚本中使用，但保留)

# stage 1: Remove repeated datas in pretrain and train set, extract audio from mp4, create mixture list
echo 'stage 1: create mixture list for 3-mix 6s'
python 1_create_mixture_list.py \
 --data_direc $data_direc \
 --C $C \
 --mix_db $mix_db \
 --train_samples $train_samples \
 --val_samples $val_samples \
 --test_samples $test_samples \
 --audio_data_direc $audio_data_direc \
 --min_length $min_length \
 --sampling_rate $sampling_rate \
 --mixture_data_list $mixture_data_list \


# stage 2: create audio mixture from list
echo 'stage 2: create mixture audios for 3-mix 6s'
python 2_create_mixture.py \
 --C $C \
 --audio_data_direc $audio_data_direc \
 --mixture_audio_direc $mixture_audio_direc \
 --mixture_data_list $mixture_data_list