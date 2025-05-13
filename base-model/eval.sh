export CUDA_VISIBLE_DEVICES=1
torchrun run_qa.py \
    --model_name_or_path /path/to/model \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 32 \
    --seed 42 \
    --max_seq_length 786 \
    --output_dir ./results/ \
    --overwrite_output_dir 
    