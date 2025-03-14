#!/bin/bash
#SBATCH -p IAI_SLURM_HGX
#SBATCH -c 16
#SBATCH -o logs/%j-BaseLlama-LIFT-LoRAMLP-QA30-C5M3-RandSeg-ShortQA.out.log
#SBATCH -e logs/%j-BaseLlama-LIFT-LoRAMLP-QA30-C5M3-RandSeg-ShortQA.err.log
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH -J BaseLlama-LIFT-LoRAMLP-QA30-C5M3-RandSeg-ShortQA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

python scripts/mp_wrapper.py \
    --script scripts/test_loogle.py \
    --num_process 4 \
    --input_file datasets/loogle/shortdep_qa.jsonl \
    --output_file outputs/BaseLlama-LIFT-LoRAMLP-QA30-C5M3-RandSeg-ShortQA.jsonl \
    --num_test 40 \
    --subprocess_args \
    --overwrite True \
    --num_syn_qa 30 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --use_chat_template False \
    --post_instruction True \
    --model_name_or_path models/Meta-Llama-3-8B \
    --use_random_segment True \
    --use_lora True \
    --lora_rank 64 \
    --lora_target_modules up_proj gate_proj down_proj \
    --load_in_4bit True \
    --involve_qa_epochs 3 \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
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
