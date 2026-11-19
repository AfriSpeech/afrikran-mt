import os
import sys
import time
import subprocess
import json

ALIGNER_BIN = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma/.venv/bin/awesome-align"
MODEL_DIR = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma/aligner/model"
LOG_FILE = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma/aligner/checkpoints_eval.log"

BENCHMARK_PAIRS = [
    ("the holy spirit gives peace", "honhom kronkron no ma asomdwoe", "twi"),
    ("water of life flows freely", "maji ya uzima yanatiririka bure", "swahili"),
    ("God loves all the children", "Onyankopon do mmofra no nyinaa", "twi"),
    ("the good news brings great joy", "habari njema huleta furaha kubwa", "swahili"),
    ("eternal life is a gift from God", "ìyè àìnípẹ̀kun jẹ́ ẹ̀bùn láti ọ̀dọ̀ Ọlọ́run", "yoruba"),
    ("almighty God created heaven and earth", "uNkulunkulu uSomandla wadala izulu nomhlaba", "zulu"),
    ("living water gives eternal life", "ruwa mai rai yana ba da rai na har abada", "hausa"),
]

def eval_checkpoint(ckpt_path: str):
    name = os.path.basename(ckpt_path)
    print(f"\n=== Evaluating Checkpoint: {name} ===")
    
    # 1. Prepare benchmark text
    tmp_input = f"/tmp/bench_{name}.txt"
    tmp_output = f"/tmp/bench_{name}.align"
    with open(tmp_input, "w", encoding="utf-8") as f:
        for src, tgt, _ in BENCHMARK_PAIRS:
            f.write(f"{src} ||| {tgt}\n")
            
    # 2. Run awesome-align
    cmd = [
        ALIGNER_BIN,
        "--model_name_or_path", ckpt_path,
        "--data_file", tmp_input,
        "--output_file", tmp_output,
        "--batch_size", "16",
        "--extraction", "softmax",
        "--align_layer", "8"
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    el = time.time() - t0
    
    if res.returncode != 0:
        print(f"Error running awesome-align on {name}: {res.stderr}")
        return None
        
    with open(tmp_output, "r", encoding="utf-8") as f:
        alignments = [l.strip() for l in f]
        
    results = []
    for (src, tgt, lang), align in zip(BENCHMARK_PAIRS, alignments):
        src_w = src.split()
        tgt_w = tgt.split()
        links = [tuple(map(int, x.split("-"))) for x in align.split() if "-" in x]
        decoded = [f"{src_w[i]}<->{tgt_w[j]}" for i, j in links if i < len(src_w) and j < len(tgt_w)]
        results.append({
            "src": src,
            "tgt": tgt,
            "lang": lang,
            "align": align,
            "decoded": decoded
        })
        print(f"[{lang.upper()}] {src} <-> {tgt}")
        print(f"       Links: {decoded}")
        
    # Append to log
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(f"\n--- {name} (Inference time: {el:.2f}s) ---\n")
        for r in results:
            lf.write(f"[{r['lang']}] {r['src']} <-> {r['tgt']}\n")
            lf.write(f"Links: {' '.join(r['decoded'])}\n")
            
    print(f"Evaluation complete in {el:.2f}s. Logged to {LOG_FILE}")
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    else:
        ckpt = os.path.join(MODEL_DIR, "checkpoint-2500")
    eval_checkpoint(ckpt)
