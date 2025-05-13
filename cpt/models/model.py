import torch
from models.modeling_gpt2_custom import GPT2ForQuestionAnswering as GPT2ForQuestionAnsweringCustom
from transformers import GPT2ForQuestionAnswering
from models.lora_modules import AdaptiveLoRALinear

def get_custom_model(custom_config):    
    model_custom = GPT2ForQuestionAnsweringCustom(custom_config)
    sd = model_custom.state_dict()
   
    # print("Custom model state dict:")
    # for k, v in sd.items():
    #     print(k, v.shape)

    model_hf = GPT2ForQuestionAnswering.from_pretrained("gpt2", torch_dtype=torch.bfloat16)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            # print(f"copying {k} with transpose")
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            # print(f"copying {k} without transpose")
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    # Set layer names for AdaptiveLoRALinear modules
    for name, module in model_custom.named_modules():
        if isinstance(module, AdaptiveLoRALinear):
            module.layer_name = name
            
    return model_custom
