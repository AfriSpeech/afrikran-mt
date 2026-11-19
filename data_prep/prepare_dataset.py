import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/mnt/volume_d2wey28/projects/africa-corpus-builder")
import africa_corpus as ac

PROJECT_DIR = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Load registry
with open(os.path.join(PROJECT_DIR, "african_languages_registry.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
REGISTRY = meta["languages"]

def is_val_verse(k: str) -> bool:
    # Hold out Gospel of John chapters 1, 2, 3 as standard multi-parallel benchmark
    return k.startswith("JHN.1.") or k.startswith("JHN.2.") or k.startswith("JHN.3.")

def clean_text(t: str) -> str:
    return " ".join(t.split()).strip()

def process_language(code: str):
    info = REGISTRY[code]
    cname = info["clean_name"]
    tag = info["tag"]
    
    # We load en inside worker or pass
    try:
        lang_obj = ac.resolve(code)
        af_verses = ac._load_verses(lang_obj)
        en_lang = ac.resolve("en")
        en_verses = ac._load_verses(en_lang)
    except Exception as e:
        return code, 0, 0, f"Error loading {code}: {e}"
        
    shared_keys = set(af_verses) & set(en_verses)
    if not shared_keys:
        return code, 0, 0, "No shared keys"
        
    train_rows = []
    val_rows = []
    
    for k in shared_keys:
        is_val = is_val_verse(k)
        target_list = val_rows if is_val else train_rows
        
        af_list = [clean_text(t) for t in af_verses[k] if len(clean_text(t)) >= 3]
        en_list = [clean_text(t) for t in en_verses[k] if len(clean_text(t)) >= 3]
        
        for en_text in en_list:
            for af_text in af_list:
                # Direction 1: English -> African
                target_list.append({
                    "instruction": f"Translate the text to {cname}:  {en_text}",
                    "response": f"{tag} {af_text}",
                    "lang": code,
                    "direction": "en2af",
                    "verse": k
                })
                # Direction 2: African -> English
                target_list.append({
                    "instruction": f"Translate the text to English:  {af_text}",
                    "response": f"<eng> {en_text}",
                    "lang": code,
                    "direction": "af2en",
                    "verse": k
                })
                
    # Save parquet for this language
    schema = pa.schema([
        ("instruction", pa.string()),
        ("response", pa.string()),
        ("lang", pa.string()),
        ("direction", pa.string()),
        ("verse", pa.string()),
    ])
    
    if train_rows:
        train_table = pa.Table.from_pylist(train_rows, schema=schema)
        pq.write_table(train_table, os.path.join(TRAIN_DIR, f"{code}.parquet"), compression="zstd")
        
    if val_rows:
        val_table = pa.Table.from_pylist(val_rows, schema=schema)
        pq.write_table(val_table, os.path.join(VAL_DIR, f"{code}.parquet"), compression="zstd")
        
    return code, len(train_rows), len(val_rows), None

def main():
    codes = sorted(REGISTRY.keys())
    print(f"Starting data preparation for {len(codes)} African languages...")
    t0 = time.time()
    
    # Pre-cache English once
    print("Pre-caching reference English versions...")
    en_lang = ac.resolve("en")
    _ = ac._load_verses(en_lang)
    print("English cached.")
    
    total_train = 0
    total_val = 0
    success = 0
    failed = 0
    
    # Process with 12 workers
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_language, code): code for code in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            code, n_train, n_val, err = fut.result()
            if err:
                print(f"[{i}/{len(codes)}] {code}: FAILED ({err})")
                failed += 1
            else:
                total_train += n_train
                total_val += n_val
                success += 1
                if i % 25 == 0 or i == len(codes):
                    elapsed = time.time() - t0
                    print(f"[{i}/{len(codes)}] Done {code} | Train: {total_train:,} | Val: {total_val:,} | Elapsed: {elapsed:.1f}s")
                    
    total_time = time.time() - t0
    print(f"\n==========================================")
    print(f"Data Preparation Complete in {total_time:.1f}s!")
    print(f"Successful languages: {success}/{len(codes)}")
    print(f"Total Train Examples (bidirectional): {total_train:,}")
    print(f"Total Val Examples (bidirectional): {total_val:,}")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Val dir: {VAL_DIR}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
