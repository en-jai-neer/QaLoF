from trainer_qa import QuestionAnsweringTrainer
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from transformers.trainer import _is_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import nested_detach
import numpy as np


class AdaptiveLoRATrainer(QuestionAnsweringTrainer):
    """
    Extension of the Hugging Face Trainer class to support
    adaptive bit-width LoRA training with knowledge distillation.
    """
    
    def __init__(
        self,
        num_cyclic_period=32,
        bit_widths=[4, 8],
        current_bit_width=None,
        *args,
        **kwargs
    ):
        """
        Initialize the AdaptiveLoRATrainer with distillation weight.
        
        Args:
            distill_weight (float): Weight for knowledge distillation loss
        """
        super().__init__(*args, **kwargs)
        self.num_cyclic_period = num_cyclic_period 
        self.bit_widths = bit_widths
        self.current_bit_width = current_bit_width
    
    def cyclic_adjust_precision(self):
        """Adjust the bit-width using a cyclic schedule."""
        num_bit_min = self.bit_widths[0]
        num_bit_max = self.bit_widths[1]
        cyclic_period = int(self.state.max_steps / self.num_cyclic_period)
        _iter = self.state.global_step
        
        # Cosine annealing for bit-width
        num_bits = np.rint(num_bit_min +
                        0.5 * (num_bit_max - num_bit_min) *
                        (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        
        return int(num_bits)
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=False):
        """
        Compute loss for the current model configuration.
        
        Args:
            model: The model to compute loss for
            inputs: Input dictionary containing model inputs
            return_outputs: Whether to return model outputs alongside loss
            
        Returns:
            Loss tensor or (loss tensor, model outputs) if return_outputs is True
        """ 
        # Handle label smoothing
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
       
        num_bits = self.cyclic_adjust_precision()
        
        if num_bits is not None:
            # print("train with current bit width", num_bits)
            outputs = model(**inputs, num_bits=num_bits)
        else:
            outputs = model(**inputs)
            
        # Save past state if it exists (for transformer models)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
            
        # Compute loss with labels if provided
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # Use the loss from model outputs
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        return (loss, outputs) if return_outputs else loss
    
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    if self.current_bit_width is not None:
                        print("eval with current bit width", self.current_bit_width)
                        outputs = model(**inputs, num_bits=self.current_bit_width)
                    else:
                        # print("eval with 8 bits")
                        outputs = model(**inputs, num_bits=8)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

 