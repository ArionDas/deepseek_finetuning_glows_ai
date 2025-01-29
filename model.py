from unsloth import FastLanguageModel
import torch

from config import CONFIG

from typing import Any


## load model
def load_model():
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CONFIG["model_name"],
        max_seq_length = CONFIG["max_seq_length"],
        dtype = None,
        load_in_4bit = CONFIG["load_in_4bit"],
    )
    
    return model, tokenizer


## get PEFT model
def get_PEFT_model(model: Any) -> Any:
    
    peft_model = FastLanguageModel.get_peft_model(
        model,
        r = 4,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
        use_rslora = False,
        loftq_config = None,
    )
    
    return peft_model