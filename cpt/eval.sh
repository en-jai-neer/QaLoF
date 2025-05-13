torchrun --nproc_per_node=8 --master_port=15001 run_qa_eval.py \
    --model_name_or_path /path/to/your/model \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 32 \
    --seed 42 \
    --output_dir ./eval_results/ \
    --overwrite_output_dir \
    --max_seq_length 786 \
    --eval_bit_width 12
