#!/bin/bash
#SBATCH -p IAI_SLURM_HGX
#SBATCH -o logs/%j-loogle-shortqa.out
#SBATCH -e logs/%j-loogle-shortqa.err
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --qos=16gpu-hgx
#SBATCH -J main_short 
#SBATCH --nodes=1 
#SBATCH --time=24:00:00
source utils.sh
prepare

python scripts/mp_wrapper.py \
    --script scripts/test_loogle_lift_random_icl_prompt.py \
    --num_process 1 \
    --input_file shortdep_qa.subset.jsonl \
    --output_file outputs/subset.OLD.dp=0.05.noCOT.jsonl \
    --subprocess_args \
    --output_dir  models/qwen-trained.debug \
    --overwrite False \
    --num_syn_qa 0 \
    --title_option 1 \
    --generator_name_or_path /ceph/home/muhan01/huggingfacemodels/Qwen2.5-32B-Instruct \
    --model_name_or_path models/qwen-trained.dp=0.05 \
    --model_max_length 32000 \
    --block_size 256 \
    --len_segment 31 \
    --len_offset 4 \
    --use_gated_memory True \
    --load_in_4bit True \
    --use_icl False \
    --use_lora False \
    --use_cot False \
    --gather_batches False \
    --involve_qa_epochs 0 \
    --num_train_epochs 0 \
    --remove_unused_columns True \
    --report_to none \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --lr_scheduler_type constant

