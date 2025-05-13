<p align="center">
    <img src="images/QaLoF Logo_ Dynamic Efficiency in LLMs.png" alt="QaLoF Logo" width="200"/>
</p>

# QaLoF: Efficient LLMs via Switchable and Dynamic Quantization

## Overview

QaLoF (Quantization-Aware-LoRa Finetuning) is a project focused on enhancing the efficiency of Large Language Models (LLMs) through quantization techniques along with LoRa finetuning. It implements switchable and dynamic quantization schemes that improve the accuracy-efficiency trade-off for LLMs, specifically targeting GPT-2 models. This work extends the LLM-QAT framework by introducing a switchable and dynamic quantization mechanism for weights, activations, and the key-value (KV) cache. It builds upon methodologies from InstantNet and CPT, incorporating switchable-precision and cyclic precision training respectively by integrating LoRA modules tuned for multiple bit-widths.

## Background

Large language models (LLMs) have demonstrated remarkable emergent abilities, but their substantial size poses significant challenges for deployment and efficiency. Among various optimization methods, quantization has emerged as a promising approach, offering a favorable accuracy-efficiency trade-off, ease of implementation, and compatibility with modern hardware.

This project investigates and implements a novel switchable and dynamic quantization scheme to further improve the performance of LLMs under resource constraints.

## Key Features

- **Per-layer Quantization**: Different bit-widths for each layer based on configurable parameters, using symmetric MinMax quantization.
- **LoRA Integration**: Multiple LoRA modules (4-bit and 8-bit) attached to linear layers in GPT-2.
- **Switchable Precision**: Ability to switch between different quantization configurations, with LoRA modules selectively activated based on user-specified bit-width configurations.
- **Cyclic Precision Training (CPT)**: Dynamic adjustment of training bit-widths throughout the finetuning process.
- **Random Precision Switch**: Dynamic quantization at inference time to improve adversarial robustness.
- **Layer-Wise Sensitivity Analysis**: To identify optimal bit-width assignments.
- **Empirical Validation**: Conducted on the SQuAD v1.1 dataset using Exact Match (EM) and F1 metrics.

## Implementation Details

The QaLoF project incorporates the following implementation details:

### Quantization Framework:
- **Weight Quantization**: Per-channel symmetric MinMax quantization.
- **Activation Quantization**: Per-token symmetric MinMax quantization.
- **KV Cache Quantization**: Per-token quantization for generation efficiency.
- **Supported Bit-widths**: 4-bit and 8-bit.

### LoRA Integration:
- Each linear layer in the GPT-2 model is augmented with two LoRA modules, one for 4-bit and one for 8-bit quantization.
- During inference, the appropriate LoRA module is activated based on the specified bit-width configuration.

### Dynamic Configuration:
- Bit-widths for each layer are specified in a user-defined configuration file. This file governs quantization settings and LoRA module selection.

### Layer-Wise Sensitivity Analysis:
- An iterative process is used to determine the optimal bit-width for each layer or attention block.
- Starting with all layers in 8-bit mode, individual layers or blocks are switched to 4-bit, and the impact on EM and F1 scores is evaluated.
- Layers are then ranked by their sensitivity to quantization, allowing for selective quantization of the least sensitive layers to maintain performance.

### Other Details:
- **Baseline Model**: GPT-2 (124M parameters) with Conv1D layers replaced by Linear layers.
- **Dataset**: SQuAD v1.1 for training and evaluation.
- **Training Strategy**: The base model is frozen; only the LoRA modules and the QA output layer are trained.
- **PTQ Library**: bitsandbytes.

## Project Structure

```
├── base_model/
│   ├── train.sh                            # Training script
│   ├── eval.sh                             # Evaluation script
│   ├── run_qa.py                           # Main script for question-answering tasks
│   ├── trainer_qa.py                       # Custom QA trainer class
│   └── utils_qa.py                         # Utility functions for QA processing
├── adv_attack/                             # Adversarial attack implementations
│   ├── text_classification.py              # Training script to finetune a pre-trained transformer model from HF
│   ├── whitebox_attack.py                  # Script to attack a finetuned model
│   ├── eval.py                             # Script to evaluate the advesarial accuracy
│   └── random_precision_inference.py       # Run RPI on a model
├── cpt/                                    # Cyclic precision training implementation
│   ├── models/                             # Files for custom gpt-2 model
│   ├── train.sh                            # Training script
│   ├── eval.sh                             # Evaluation script
│   ├── run_qa.py                           # Main script for question-answering tasks
│   ├── trainer_qa.py                       # Custom QA trainer class
│   └── utils_qa.py                         # Utility functions for QA processing
└── switch_precision/                       # Switchable precision implementation
    ├── models/                             # Files for custom gpt-2 model
    ├── train.sh                            # Training script
    ├── eval.sh                             # Evaluation script
    ├── run_qa.py                           # Main script for question-answering tasks
    ├── trainer_qa.py                       # Custom QA trainer class
    ├── utils_qa.py                         # Utility functions for QA processing
    ├── adaptive_lora_trainer.py            # Custom trainer to implement QaLoF
    └── run_qa_eval.py                      # Eval to run eval
```

## Usage

### Training

To train the model (example from base_model directory, adapt for other modules like cpt or switch_precision which have their own training scripts):

```bash
# From the base_model directory
bash train.sh
```

The training script includes the following key parameters (example from base_model):

```bash
torchrun --nproc_per_node=8 --master_port=15001 run_qa.py \
    --model_name_or_path /path/to/model \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --seed 42 \
    --learning_rate 3e-03 \
    --logging_strategy steps \
    --logging_steps 10 \
    --lr_scheduler_type cosine \
    --max_seq_length 786 \
    --bf16 False \
    --fp16 False \
    --max_steps 1000 \
    --output_dir ./results/your-output-directory \
    --overwrite_output_dir \
    --eval_strategy steps \
    --eval_steps 250 \
    --save_strategy steps \
    --save_steps 250 \
    --lora_r 8 \
    --lora_alpha 16 \
    --bit_widths 4
```

For switchable precision the training details are - 1000 iterations, AdamW optimizer, LoRA hyperparameters $r=8, \alpha=16$, and training on 8 GPUs with a batch size of 16 per device (effective batch size of 128). For CPT, the learning rate was 2e-3 with 32 cyclic periods and a bit range of 4-8.

### Evaluation

To evaluate the model (example from base_model directory, adapt for other modules):

```bash
# From the base_model directory
bash eval.sh
```

The evaluation script includes the following key parameters (example from base_model):

```bash
torchrun run_qa_eval.py \
    --model_name_or_path /path/to/model/checkpoint \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 32 \
    --seed 42 \
    --max_seq_length 786 \
    --output_dir ./results/your-evaluation-directory \
    --overwrite_output_dir 
```

## Requirements

- PyTorch
- Transformers (Hugging Face)
- SQuAD dataset
- CUDA-compatible GPU (for efficient training, 8 GPUs used in experiments)
- bitsandbytes library

## Results and Insights

The project demonstrates how different quantization strategies affect model performance, particularly:

### Impact of per-layer bit-width configurations:
- Later transformer blocks (specifically blocks 8-11) exhibit greater resilience to quantization, while early and middle blocks are more sensitive.
- Within attention layers, the attn.c_proj sub-layer is more quantization-tolerant than attn.c_attn.
- In MLP layers, mlp.c_proj is more robust than mlp.c_fc, particularly in earlier blocks.
- MLP layers are generally more sensitive to reduced precision compared to attention layers. Quantizing all MLP or all attention layers individually to 4 bits can lead to greater performance degradation than quantizing the entire model uniformly to 4-bit precision.
- An optimal hybrid configuration was identified: quantize attn.c_proj in blocks 0-3, no quantization in middle blocks (4-7) and block 8, and quantize all layers in blocks 9-11. This configuration results in approximately 25% of the model operating at 4-bit and 75% at 8-bit, achieving an EM score of 60.98 and F1 score of 72.31.

### Benefits of Cyclic Precision Training (CPT):
- CPT did not yield performance improvements over static precision training for 8-bit quantization in the conducted experiments.
- However, a key advantage of CPT is that the model requires only a single training run to support a broad range of quantization bit-widths at inference time.
- Performance at 6-bit (EM: 63.05, F1: 74.26) closely matched 8-bit (EM: 63.54, F1: 74.70), and 7-bit (EM: 63.77, F1: 74.85) surpassed 8-bit performance. These configurations showed performance comparable to full-precision (FP32) and 8-bit quantization with static precision training (EM: 63.83, F1: 75.08).

### Effect of dynamic quantization on adversarial robustness:
- Experiments using Gradient-Based Distributional Attack (GBDA) on a pretrained FP32 GPT-2 model (adversarial accuracy 3%) were conducted.
- Applying post-training quantization showed a modest improvement: 8-bit GPT-2 adversarial accuracy increased to 4%, and 4-bit GPT-2 adversarial accuracy increased to 6%.
- While there's a trend consistent with findings that lower-bit inference improves adversarial robustness (Double Win Quant), the magnitude of this improvement was small in these tests.

## Future Work

Future directions for this project include:
- Automating sensitivity profiling using gradient-based heuristics.
- Incorporating more granular quantization schemes (e.g., per-head, per-token).
- Extending support to larger models and broader NLP tasks.
- Addressing limitations in the current training setup, such as incorporating higher-precision teacher logits for distillation to potentially improve low bit-width student model performance.
- Exploring stochastic bit-width sampling during training to enhance robustness across diverse quantization schemes.
- Investigating contrastive representation alignment and Quantization-Aware Knowledge Distillation (QAKD) for intermediate representations.
- Adopting curriculum quantization, progressively introducing lower bit-widths during training..

## Acknowledgments

- This project builds upon Hugging Face's transformers library examples for Question Answering.
- Methodology inspired by recent advances in efficient LLM deployment techniques, including work on LLM-QAT, InstantNet, CPT, and Double-Win Quant.