
import os

from datasets import load_dataset, Dataset


def process_gpqa(raw_data_path: str, processed_data_path: str, name: str = "gpqa_main"):
    """
    Read GPQA data set, only keep Question and Correct Answer fields, and save to new path
    """
    raw_dataset = load_dataset(raw_data_path, name=name, split="train")
    print(f"Before processing: {len(raw_dataset)}")
    
    processed_data = []
    for item in raw_dataset:
        processed_data.append({
            "question": item["Question"],
            "answer": item["Correct Answer"]
        })
    
    processed_dataset = Dataset.from_list(processed_data)
    print(f"After processing: {len(processed_dataset)}")
    
    os.makedirs(processed_data_path, exist_ok=True)
    
    processed_dataset.to_parquet(os.path.join(processed_data_path, "train.parquet"))
    print(f"Saved to: {os.path.join(processed_data_path, 'train.parquet')}")
    
    return processed_dataset

if __name__ == "__main__":
    raw_data_path = "/mnt/data/yuchang/datasets/gpqa"
    for subset in ["main", "diamond"]:
        processed_data_path = f"/cpfs/data/tir/data/gpqa/{subset}"
        
        processed_dataset = process_gpqa(raw_data_path, processed_data_path, name=f"gpqa_{subset}")
        
        print("\nExample:")
        print(processed_dataset[0])

