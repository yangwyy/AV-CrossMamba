#!/bin/sh

gpu_id=4,5
continue_from=

# 检查 logs 目录是否存在，如果不存在则创建
if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ -z ${continue_from} ]; then
	log_name='avaNet_'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=1236 \
main.py \
\
--log_name $log_name \
\
--batch_size 2 \
--audio_direc '/home/wangcanyang/vox2clean_audio/' \
--visual_direc '/home/wangcanyang/vox2lip/' \
--mix_lst_path '/home/wangcanyang/MuSE/data/voxceleb2-800/mixture_data_list_2mix.csv' \
--mixture_direc '/home/wangcanyang/vox2_2mix/2_mix/' \
--C 2 \
--epochs 100 \
\
--use_tensorboard 1 \
2>&1 | tee -a logs/$log_name/console.txt
