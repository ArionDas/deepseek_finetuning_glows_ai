from datasets import load_dataset

from standardize_shareGPT import standardize_shareGPT


def get_dataset():
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    print("Dataset Loaded!!")
    print("Dataset Column Names: ")
    print(dataset.column_names)

    dataset = standardize_shareGPT(dataset)
    
    return dataset