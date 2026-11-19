import os
import sys
import glob
import random
import pyarrow.parquet as pq

DATA_DIR = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma/data/train"
ALIGN_DIR = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma/aligner"
os.makedirs(ALIGN_DIR, exist_ok=True)

train_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
print(f"Sampling parallel data across {len(train_files)} language shards...")

train_lines = []
eval_lines = []

random.seed(42)

for sf in train_files:
    try:
        table = pq.read_table(sf, columns=["instruction", "response", "direction"])
        # Filter to en2af
        insts = table["instruction"]
        resps = table["response"]
        dirs = table["direction"]
        
        pairs = []
        for i in range(len(table)):
            if dirs[i].as_py() == "en2af":
                inst = insts[i].as_py()
                resp = resps[i].as_py()
                # Extract clean English
                if ":  " in inst:
                    en_text = inst.split(":  ", 1)[1].strip()
                else:
                    en_text = inst.strip()
                # Extract clean African (strip tag)
                parts = resp.strip().split(" ", 1)
                af_text = parts[1].strip() if len(parts) > 1 else resp.strip()
                
                if len(en_text) >= 5 and len(af_text) >= 5:
                    pairs.append(f"{en_text.lower()} ||| {af_text.lower()}\n")
                    
        if pairs:
            # Sample up to 500 pairs per language
            sample_k = min(500, len(pairs))
            sampled = random.sample(pairs, sample_k)
            # 95% train, 5% eval
            split_idx = int(sample_k * 0.95)
            train_lines.extend(sampled[:split_idx])
            eval_lines.extend(sampled[split_idx:])
    except Exception as e:
        print(f"Error on {sf}: {e}")

random.shuffle(train_lines)
random.shuffle(eval_lines)

train_path = os.path.join(ALIGN_DIR, "train_align.txt")
eval_path = os.path.join(ALIGN_DIR, "eval_align.txt")

with open(train_path, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(eval_path, "w", encoding="utf-8") as f:
    f.writelines(eval_lines)

print(f"Exported {len(train_lines):,} training pairs to {train_path}")
print(f"Exported {len(eval_lines):,} evaluation pairs to {eval_path}")
