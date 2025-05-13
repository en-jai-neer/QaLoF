torchrun --nproc_per_node=8 --master_port=15002 run_qa_eval.py \
    --model_name_or_path /path/to/your/model \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --seed 42 \
    --output_dir ./eval_results/last-layers-7 \
    --overwrite_output_dir \
    --max_seq_length 786 \
    --sp_config ./utils/sp_config.json
