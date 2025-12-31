import os
import pandas as pd

def prepare_dataset():
    files = [
        "https://huggingface.co/datasets/ageron/tatoeba_mt_train/resolve/main/eng-spa/test-00000-of-00001.parquet?download=true",
        "https://huggingface.co/datasets/ageron/tatoeba_mt_train/resolve/main/eng-spa/validation-00000-of-00001.parquet?download=true"
    ]
    cols = ["source_text", "target_text"]
    data_name = "eng_spa.parquet"

    # Read, concatenate, shuffle, and reset index
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], axis=0)\
        .sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to parquet
    df.to_parquet(data_name)
    print(f"Data saved to {os.path.join(os.getcwd(), data_name)}")

if __name__ == "__main__":
    prepare_dataset()