import re
import numpy as np
from typing import Optional, List
from datasets import load_dataset, concatenate_datasets, Audio, Dataset
from huggingface_hub import login
import os

KEEP_CHARS = " абвгдеёжзийклмноөпрстуүфхцчшъыьэюя"
DEFAULT_SAMPLING_RATE = 16_000
MAX_LENGTH = 448

def prepare_dataset(
    dataset_name: str,
    config: str,
    split: str,
    keep_chars: str = KEEP_CHARS,
) -> Dataset:
    """Load and prepare a single dataset."""
    # Ensure we're authenticated with Hugging Face
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    
    ds = load_dataset(dataset_name, config, split=split)
    
    # Standardize column names
    text_columns = {'sentence', 'transcription', 'text'}
    text_col = list(set(ds.column_names).intersection(text_columns))[0]
    if text_col != 'text':
        ds = ds.rename_column(text_col, 'text')

    # Remove unnecessary columns
    keep_columns = {'audio', 'text'}
    remove_cols = list(set(ds.column_names) - keep_columns)
    ds = ds.remove_columns(remove_cols)

    # Ensure audio is in correct format
    ds = ds.cast_column("audio", Audio(sampling_rate=DEFAULT_SAMPLING_RATE))
    
    # Add preprocessing
    ds = ds.map(
        lambda x: {"text": preprocess_text(x["text"], keep_chars)},
        desc="Preprocessing text"
    )
    
    return ds

def preprocess_text(text: str, keep_chars: str) -> str:
    """Preprocess text by lowercasing and removing unwanted characters."""
    return re.sub(f"[^{keep_chars}]", "", text.lower())

def load_local_common_voice(
    data_dir: str,
    split: str,
    keep_chars: str = KEEP_CHARS,
    chunk_size: int = 500  # Process dataset in smaller chunks
) -> Dataset:
    """Load Common Voice dataset from local directory."""
    import pandas as pd
    
    # Read TSV file
    tsv_path = os.path.join(data_dir, f"{split}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Dataset file not found: {tsv_path}")
        
    print(f"Loading dataset from {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Verify required columns exist
    required_columns = ['path', 'sentence']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    
    # Create dataset dictionary
    clips_dir = os.path.join(data_dir, 'clips')
    if not os.path.exists(clips_dir):
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    
    # Process dataset in chunks to reduce memory usage
    datasets = []
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i + chunk_size]
        dataset_dict = {
            'audio': [os.path.join(clips_dir, path) for path in chunk_df['path']],
            'text': [preprocess_text(text, keep_chars) for text in chunk_df['sentence']]
        }
        
        # Verify first chunk's audio files exist
        if i == 0:
            for audio_path in dataset_dict['audio'][:5]:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Processing chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}")
        chunk_ds = Dataset.from_dict(dataset_dict)
        chunk_ds = chunk_ds.cast_column("audio", Audio(sampling_rate=DEFAULT_SAMPLING_RATE))
        datasets.append(chunk_ds)
    
    if not datasets:
        raise ValueError("No data loaded from dataset")
        
    # Concatenate all chunks
    print("Concatenating chunks...")
    final_dataset = concatenate_datasets(datasets)
    print(f"Final dataset size: {len(final_dataset)} examples")
    
    return final_dataset

def prepare_datasets(
    train_datasets: str,
    eval_datasets: Optional[str] = None,
    keep_chars: str = KEEP_CHARS,
    local_data_dir: Optional[str] = None
) -> tuple[Dataset, Optional[Dataset]]:
    """Prepare multiple datasets for training and evaluation."""
    
    if local_data_dir:
        # Use local dataset
        train_ds = load_local_common_voice(local_data_dir, "train", keep_chars)
        if eval_datasets:
            eval_ds = load_local_common_voice(local_data_dir, "test", keep_chars)
        else:
            eval_ds = None
        return train_ds, eval_ds

    def load_datasets(dataset_string: str) -> List[Dataset]:
        if not dataset_string:
            return []
        
        datasets = []
        for dataset_spec in dataset_string.split(','):
            name, config, splits = dataset_spec.split('|')
            config = config if config else None
            for split in splits.split('+'):
                ds = prepare_dataset(name, config, split, keep_chars)
                datasets.append(ds)
        return datasets

    train_ds_list = load_datasets(train_datasets)
    eval_ds_list = load_datasets(eval_datasets) if eval_datasets else []

    train_ds = concatenate_datasets(train_ds_list) if train_ds_list else None
    eval_ds = concatenate_datasets(eval_ds_list) if eval_ds_list else None

    return train_ds, eval_ds

