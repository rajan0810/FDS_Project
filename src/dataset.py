import os
import json
import requests
from pathlib import Path
from .config import CONFIG, BASE_DIR

CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CACHE_PATH = CACHE_DIR / "qasper_val.json"

def load_qasper_dataset(split="validation"):
    if DATASET_CACHE_PATH.exists():
        with open(DATASET_CACHE_PATH, "r") as f:
            data = json.load(f)
            # The original load_dataset returns a list of dicts.
            # QASPER json is a dict mapping paper_id to paper_data.
            # We'll flatten it to a list of dicts with 'id' added to match the previous structure.
            dataset = [{"id": k, **v} for k, v in data.items()]
            return dataset
    
    import tarfile
    import io
    print(f"Downloading QASPER {split} split JSON from AWS S3...")
    
    # QASPER Dev set URL (Validation)
    url = "https://qasper-dataset.s3-us-west-2.amazonaws.com/qasper-train-dev-v0.1.tgz"
    if split == "validation":
        json_filename = "qasper-dev-v0.1.json"
    else:
        json_filename = "qasper-train-v0.1.json"
        
    response = requests.get(url)
    response.raise_for_status()
    
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        f = tar.extractfile(json_filename)
        data = json.loads(f.read().decode('utf-8'))
        
    with open(DATASET_CACHE_PATH, "w") as f:
        json.dump(data, f)
        
    dataset = [{"id": k, **v} for k, v in data.items()]
    return dataset
