from unsloth import to_sharegpt
from unsloth import standardize_sharegpt

from typing import Any

def standardize_shareGPT(dataset: Any) -> Any:
    
    dataset = to_sharegpt(
        dataset,
        merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
        output_column_name = "output",
        conversation_extension = 3, # Select more to handle longer conversations
    )
    
    dataset = standardize_sharegpt(dataset)
    
    return dataset