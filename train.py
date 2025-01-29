from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bflota16_supported

from typing import Any

from config import CONFIG
from model import load_model, get_PEFT_model
from data_prep import get_dataset


def model_trainer(model: Any, tokenizer: Any, dataset: Any) -> Any:
    
    model_trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = CONFIG["max_seq_length"],
        dataset_num_proc = 2,
        packing = False,
        
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = CONFIG["learning_rate"],
            fp16 = not is_bflota16_supported(),
            bf16 = is_bflota16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )
    
    return model_trainer


def train():
        
        ## load model
        model, tokenizer = load_model()
        
        ## get PEFT model
        peft_model = get_PEFT_model(model)
        
        ## get dataset
        dataset = get_dataset()
        
        ## model trainer
        model_trainer = model_trainer(peft_model, tokenizer, dataset)
        
        ## train model
        model_trainer.train()
        
        return