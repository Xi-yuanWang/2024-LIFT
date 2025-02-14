#!/bin/bash
#SBATCH -p IAI_SLURM_HGX
#SBATCH -o logs/%j-LongBench-MLPGate-rICL-QA10-C3M5-gov_report.out.log
#SBATCH -e logs/%j-LongBench-MLPGate-rICL-QA10-C3M5-gov_report.err.log
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH -J gov_report
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

python scripts/test_longbench_normal_syn_qa.py \
    --subtask_name gov_report \
    --input_dir datasets/longbench_sampling \
    --output_path outputs/debug.jsonl \
    --overwrite True \
    --num_syn_qa 0 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --use_icl True \
    --model_name_or_path models/MLPGate-Llama-3-8B-Instruct \
    --model_max_length 7800 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --use_gated_memory True \
    --load_in_4bit True \
    --gather_batches True \
    --involve_qa_epochs 0 \
    --num_train_epochs 4 \
    --learning_rate 2e-5 \
    --remove_unused_columns False \
    --report_to none \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --weight_decay 1e-4 \
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
