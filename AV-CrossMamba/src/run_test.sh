 CUDA_VISIBLE_DEVICES=5 python evaluate.py \
   --mix_lst_path   '/home/wangcanyang/MuSE/data/voxceleb2-800/mixture_data_list_4mix_6s.csv' \
   --audio_direc    '/home/wangcanyang/vox2clean_audio/' \
   --visual_direc   '/home/wangcanyang/vox2lip/' \
   --mixture_direc  '/home/wangcanyang/vox2_3mix/4_mix_6s/' \
   --continue_from  '/home/wangcanyang/MuSE/src/logs/avaNet_2025-11-11(17_56_37)/' \
   --C 4 --N 256 \
   --save_est_dir '/home/wangcanyang/MuSE/estimates-4mix-6s/'
##!/bin/bash
#
## =======================================================
## 1. 配置参数
## =======================================================
#
## 定义要运行的序列时长列表 (秒). 您可以根据需要修改此列表。
#DURATIONS=(11 12 13 14 15)
#
## 定义固定不变的基路径
#BASE_PATH="/home/wangcanyang/MuSE"
#AUDIO_DIREC="/home/wangcanyang/vox2clean_audio/"
#VISUAL_DIREC="/home/wangcanyang/vox2lip/"
#
## 模型权重路径。注意：移除了末尾的斜杠，让os.path.join更安全
#LOGS_DIR="${BASE_PATH}/src/logs/avaNet_2025-10-26(08_51_37)"
#
## 结果保存的根目录
#ESTIMATES_BASE="/home/wangcanyang/MuSE/estimates_multi_duration_muse"
#
## 最终结果汇总文件
#SUMMARY_FILE="evaluation_summary_multi_duration.txt"
#
## 设置要使用的 GPU (与您的命令行一致)
#export CUDA_VISIBLE_DEVICES=5
#
## 清空或创建汇总文件
#echo "--- AVSS Mamba Evaluation Summary ---" > "${SUMMARY_FILE}"
#echo "Model Checkpoint: ${LOGS_DIR}" >> "${SUMMARY_FILE}"
#echo "Evaluated Durations: ${DURATIONS[@]}" >> "${SUMMARY_FILE}"
#echo "-------------------------------------" >> "${SUMMARY_FILE}"
#
## =======================================================
## 2. 循环执行评估
## =======================================================
#
#for SECONDS in "${DURATIONS[@]}"; do
#
#    echo ""
#    echo "=================================================="
#    echo "Starting evaluation for duration: ${SECONDS}s"
#    echo "=================================================="
#
#    # 动态构造文件和目录路径
#    # e.g., /home/wangcanyang/MuSE/data/voxceleb2-800/mixture_data_list_2mix_5s.csv
#    MIX_LST_PATH="${BASE_PATH}/data/voxceleb2-800/mixture_data_list_2mix_${SECONDS}s.csv"
#
#    # e.g., /home/wangcanyang/vox2_2mix/2_mix_5s/
#    MIXTURE_DIREC="/home/wangcanyang/vox2_2mix/2_mix_${SECONDS}s/"
#
#    # e.g., /home/wangcanyang/MuSE/estimates_multi_duration/5s/
#    SAVE_EST_DIR="${ESTIMATES_BASE}/${SECONDS}s/"
#
#    # 检查数据文件是否存在
#    if [ ! -f "${MIX_LST_PATH}" ]; then
#        echo "Warning: Mix list file not found for ${SECONDS}s: ${MIX_LST_PATH}" | tee -a "${SUMMARY_FILE}"
#        continue
#    fi
#
#    # 直接执行命令，避免构建 COMMAND 字符串变量导致引号问题
#    python evaluate.py \
#        --mix_lst_path "${MIX_LST_PATH}" \
#        --audio_direc "${AUDIO_DIREC}" \
#        --visual_direc "${VISUAL_DIREC}" \
#        --mixture_direc "${MIXTURE_DIREC}" \
#        --continue_from "${LOGS_DIR}" \
#        --C 2 --N 256 \
#        --save_est_dir "${SAVE_EST_DIR}" 2>&1 | tee -a "${SUMMARY_FILE}"
#
#    # 在汇总文件中添加分隔符
#    echo "" >> "${SUMMARY_FILE}"
#
#done
#
#echo ""
#echo "=================================================="
#echo "All evaluations finished. Results summarized in ${SUMMARY_FILE}"
#echo "=================================================="