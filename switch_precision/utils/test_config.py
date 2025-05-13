import copy
from itertools import product

def generate_configs(base_config=None, experiment_type="layer_wise"):
    """Generate different selective precision configs for experimentation."""
    configs = []
    
    if base_config is None:
        # Default base config: 8-bit for attention layers, 4-bit for MLP layers
        base_config = {}
        for layer_idx in range(12):  # GPT-2 has 12 layers
            # Attention layers
            base_config[f"transformer.h.{layer_idx}.attn.c_attn"] = 8
            base_config[f"transformer.h.{layer_idx}.attn.c_proj"] = 8
            # MLP layers
            base_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = 8
            base_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = 8
    
    if experiment_type == "layer_wise":
        # Experiment with different bit-widths for specific layers
        bit_options = [4, 8]
        
        for layer_idx in range(12):
            for attn_bits, mlp_bits in product(bit_options, bit_options):
                # Skip if both match base config for this layer
                if (base_config.get(f"transformer.h.{layer_idx}.attn.c_attn") == attn_bits and 
                    base_config.get(f"transformer.h.{layer_idx}.mlp.c_fc") == mlp_bits):
                    continue
                
                # Create a new config by copying the base config
                new_config = copy.deepcopy(base_config)
                
                # Modify only the specified layer
                new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = attn_bits
                new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = attn_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = mlp_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = mlp_bits
                
                configs.append({
                    "config": new_config,
                    "description": f"Layer {layer_idx}: Attention={attn_bits}-bit, MLP={mlp_bits}-bit"
                })
    
    elif experiment_type == "uniform":
        # Experiment with uniform bit-widths across all layers
        bit_options = [4, 8]
        
        for bits in bit_options:
            new_config = {}
            for layer_idx in range(12):
                # Set all layers to the same bit-width
                new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = bits
                new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = bits
            
            configs.append({
                "config": new_config,
                "description": f"Uniform {bits}-bit precision across all layers"
            })
    
    elif experiment_type == "attention_vs_mlp":
        # Experiment with different bit-widths for attention vs MLP layers
        attn_options = [4, 8]
        mlp_options = [4, 8]
        
        for attn_bits, mlp_bits in product(attn_options, mlp_options):
            # Skip if matches base config pattern
            skip = True
            for layer_idx in range(12):
                if (base_config.get(f"transformer.h.{layer_idx}.attn.c_attn") != attn_bits or
                    base_config.get(f"transformer.h.{layer_idx}.mlp.c_fc") != mlp_bits):
                    skip = False
                    break
            if skip:
                continue
                
            new_config = {}
            for layer_idx in range(12):
                # Set attention and MLP layers to different bit-widths
                new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = attn_bits
                new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = attn_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = mlp_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = mlp_bits
            
            configs.append({
                "config": new_config,
                "description": f"Attention={attn_bits}-bit, MLP={mlp_bits}-bit across all layers"
            })
    
    elif experiment_type == "depth_based":
        # Experiment with different bit-widths depending on layer depth
        # Early layers (0-3), Middle layers (4-7), Late layers (8-11)
        bit_options = [4, 8]
        
        for early_bits, mid_bits, late_bits in product(bit_options, bit_options, bit_options):
            new_config = {}
            
            # Early layers (0-3)
            for layer_idx in range(0, 4):
                new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = early_bits
                new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = early_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = early_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = early_bits
            
            # Middle layers (4-7)
            for layer_idx in range(4, 8):
                new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = mid_bits
                new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = mid_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = mid_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = mid_bits
            
            # Late layers (8-11)
            for layer_idx in range(8, 12):
                new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = late_bits
                new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = late_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = late_bits
                new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = late_bits
            
            configs.append({
                "config": new_config,
                "description": f"Early={early_bits}-bit, Middle={mid_bits}-bit, Late={late_bits}-bit"
            })
    
    elif experiment_type == "custom":
        # Add specific custom configurations here
        # Example: High precision in early and late layers, low precision in middle layers
        custom_config = copy.deepcopy(base_config)
        # Early layers (0-3): 8-bit
        
        for layer_idx in range(0, 1):
            custom_config[f"transformer.h.{layer_idx}.attn.c_attn"] = 4
            custom_config[f"transformer.h.{layer_idx}.attn.c_proj"] = 4
            custom_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = 4
            custom_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = 4
            
        for layer_idx in range(1, 4):
            custom_config[f"transformer.h.{layer_idx}.attn.c_attn"] = 8
            custom_config[f"transformer.h.{layer_idx}.attn.c_proj"] = 8
            custom_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = 8
            custom_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = 8
        
        # Middle layers (4-7): 2-bit
        for layer_idx in range(4, 7):
            custom_config[f"transformer.h.{layer_idx}.attn.c_attn"] = 8
            custom_config[f"transformer.h.{layer_idx}.attn.c_proj"] = 8
            custom_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = 8
            custom_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = 8
        
        # Late layers (8-11): 8-bit
        for layer_idx in range(7, 12):
            custom_config[f"transformer.h.{layer_idx}.attn.c_attn"] = 4
            custom_config[f"transformer.h.{layer_idx}.attn.c_proj"] = 4
            custom_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = 8
            custom_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = 8
        
        configs.append({
            "config": custom_config,
            "description": "High precision in early/late layers, low precision in middle layers"
        })
    
    elif experiment_type == "attn_only":        
        for layer_idx in range(12):
            # Create a new config by copying the base config
            new_config = copy.deepcopy(base_config)
            
            # Modify only the specified layer
            new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = 4
            new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = 4
            
            configs.append({
                "config": new_config,
                "description": f"Layer {layer_idx}: Attention=4-bit, MLP=8-bit"
            })
            
    elif experiment_type == "c_attn_only":        
        for layer_idx in range(12):
            # Create a new config by copying the base config
            new_config = copy.deepcopy(base_config)
            
            # Modify only the specified layer
            new_config[f"transformer.h.{layer_idx}.attn.c_attn"] = 4
            
            configs.append({
                "config": new_config,
                "description": f"Layer {layer_idx}: c_Attention=4-bit, MLP=8-bit"
            })
            
    elif experiment_type == "c_proj_attn_only":        
        for layer_idx in range(12):
            # Create a new config by copying the base config
            new_config = copy.deepcopy(base_config)
            
            # Modify only the specified layer
            new_config[f"transformer.h.{layer_idx}.attn.c_proj"] = 4
            
            configs.append({
                "config": new_config,
                "description": f"Layer {layer_idx}: c_Attention=4-bit, MLP=8-bit"
            })
            
    elif experiment_type == "mp_only":        
        for layer_idx in range(12):
            # Create a new config by copying the base config
            new_config = copy.deepcopy(base_config)
            
            # Modify only the specified layer
            new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = 4
            new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = 4
            
            configs.append({
                "config": new_config,
                "description": f"Layer {layer_idx}: Attention=4-bit, MLP=8-bit"
            })
            
    elif experiment_type == "mlp_c_fc_only":        
        for layer_idx in range(12):
            # Create a new config by copying the base config
            new_config = copy.deepcopy(base_config)
            
            # Modify only the specified layer
            new_config[f"transformer.h.{layer_idx}.mlp.c_fc"] = 4
            
            configs.append({
                "config": new_config,
                "description": f"Layer {layer_idx}: Attention=8-bit, MLP_c_fc=4-bit"
            })
            
    elif experiment_type == "mlp_c_proj_only":        
        for layer_idx in range(12):
            # Create a new config by copying the base config
            new_config = copy.deepcopy(base_config)
            
            # Modify only the specified layer
            new_config[f"transformer.h.{layer_idx}.mlp.c_proj"] = 4
            
            configs.append({
                "config": new_config,
                "description": f"Layer {layer_idx}: Attention=8-bit, MLP_c_proj=4-bit"
            })
    
    return configs
