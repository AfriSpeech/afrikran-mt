"""
build_nkrane_dataset_neural.py — Pan-African Dual-Sided Neural Alignment & Nkrane Preserved Bracket Generator.

Implements the official Nkrane-MT Dual-Sided Bracket Contract across 693 African Languages:
1. Extracts concrete entities & complex nominals (excluding abstract concepts via WordNet).
2. Uses fine-tuned awesome-align (AfriSpeech/afro-awesome-align) for neural token links.
3. Projects brackets on BOTH SIDES in linear O(N) token time:
   - En->Af Input:    "Translate the text to {Lang}:  The <president of Ghana> visited the <Ministry of Finance>..."
   - En->Af Response: "<{tag}> <president of Ghana> no kɔsra <Ministry of Finance> no..."
   - Af->En Input:    "Translate the text to English:  <Ghana manpanin> no kɔsra <Sikasɛm Asoeɛ> no..."
   - Af->En Response: "<eng> The <Ghana manpanin> visited the <Sikasɛm Asoeɛ>..."
4. Caches English nominal extractions in memory with multi-process nlp.pipe across all 693 languages.
5. Exports:
   - data/train_bracketed_neural/{code}.parquet
   - data/val_bracketed_neural/{code}.parquet
   - dictionaries_neural/{code}_dictionary.tsv.gz
6. Periodic 30-minute atomic folder sync to Hugging Face (AfriSpeech/afrikrane-mt).
"""

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from huggingface_hub import HfApi
    HF_API = HfApi()
except ImportError:
    HF_API = None

sys.path.insert(0, "/mnt/volume_d2wey28/projects/africa-corpus-builder")
import africa_corpus as ac

from afrikrane.extractor import is_abstract_word, get_nlp

PROJECT_DIR = "/mnt/volume_d2wey28/projects/afrikrane-mt"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_OUT = os.path.join(DATA_DIR, "train_bracketed_neural")
VAL_OUT = os.path.join(DATA_DIR, "val_bracketed_neural")
DICT_OUT = os.path.join(PROJECT_DIR, "dictionaries_neural")

ALIGNER_BIN = "/mnt/volume_d2wey28/projects/afrikrane-mt/.venv/bin/awesome-align"
DEFAULT_MODEL = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma/aligner/model_final"
DEFAULT_HF_REPO = "AfriSpeech/afrikrane-mt"

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(VAL_OUT, exist_ok=True)
os.makedirs(DICT_OUT, exist_ok=True)

REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "african_languages_registry.json"
)
if not os.path.exists(REGISTRY_PATH):
    REGISTRY_PATH = os.path.join(PROJECT_DIR, "data", "african_languages_registry.json")

with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)
REGISTRY = META["languages"]

NAME_TO_CODE = {v["clean_name"].lower(): k for k, v in REGISTRY.items()}
RAW_TO_CODE = {v["raw_name"].lower(): k for k, v in REGISTRY.items()}

ISO_2_TO_3 = {
    "tw": "twi", "yo": "yor", "ha": "hau", "ig": "ibo", "sw": "swh", "zu": "zul",
    "xh": "xho", "af": "afr", "am": "amh", "so": "som", "om": "orm", "ti": "tir",
    "sn": "sna", "ny": "nya", "st": "sot", "tn": "tsn", "ts": "tso", "ve": "ven",
    "ss": "ssw", "nr": "nbl", "nd": "nde", "sg": "sag", "lg": "lug", "ln": "lin",
    "rw": "kin", "rn": "run", "mg": "mlg", "wo": "wol", "ff": "ful", "ee": "ewe",
    "bm": "bam", "kg": "kon", "lu": "lub", "kj": "kua", "ng": "ndo", "kr": "knc"
}

# Global in-memory cache for parsed English phrases: en_text -> (tokens, phrases, spans)
EN_PARSE_CACHE: Dict[str, Tuple[List[str], List[str], List[Tuple[int, int, str]]]] = {}


def resolve_3letter_code(code: str, name: str) -> str:
    c_low = code.lower().strip()
    if c_low in REGISTRY:
        return c_low
    if c_low in ISO_2_TO_3:
        return ISO_2_TO_3[c_low]
    n_low = name.lower().strip()
    if n_low in NAME_TO_CODE:
        return NAME_TO_CODE[n_low]
    if n_low in RAW_TO_CODE:
        return RAW_TO_CODE[n_low]
    for clean_n, k in NAME_TO_CODE.items():
        if clean_n in n_low or n_low in clean_n:
            return k
    return c_low


def clean_txt(t: str) -> str:
    return " ".join(t.split()).strip()


def is_val_key(k: str) -> bool:
    return k.startswith("JHN.1.") or k.startswith("JHN.2.") or k.startswith("JHN.3.") or "#val" in k


def populate_en_cache(en_texts: List[str], nlp, n_process: int = 8):
    """Batch parses uncached English sentences directly from doc stream without re-parsing."""
    uncached = [t for t in set(en_texts) if t not in EN_PARSE_CACHE]
    if not uncached:
        return

    for doc in nlp.pipe(uncached, batch_size=1000, n_process=n_process):
        text = doc.text
        tokens = [t.text for t in doc]
        phrases = []
        spans = []

        for chunk in doc.noun_chunks:
            root = chunk.root
            if is_abstract_word(root.text):
                continue
            curr_start = chunk.start
            curr_end = chunk.end
            while curr_start < curr_end and doc[curr_start].is_stop and doc[curr_start] != root:
                curr_start += 1
            if curr_start < curr_end:
                phrase = doc[curr_start:curr_end].text.strip()
                if phrase:
                    phrases.append(phrase)
                    spans.append((curr_start, curr_end, phrase))

        EN_PARSE_CACHE[text] = (tokens, phrases, spans)


class PeriodicSyncThread(threading.Thread):
    """Batches Hugging Face uploads every 30 minutes to respect API rate limits."""
    def __init__(self, hf_repo: Optional[str], sync_interval_secs: int = 1800):
        super().__init__(daemon=True)
        self.hf_repo = hf_repo
        self.interval = sync_interval_secs
        self.running = True

    def run(self):
        while self.running:
            time.sleep(self.interval)
            self.sync_now()

    def sync_now(self):
        if not self.hf_repo or not HF_API:
            return
        t0 = time.time()
        print(f"\n>> [HF BATCH SYNC] Starting periodic batch sync to {self.hf_repo}...", flush=True)
        try:
            if os.path.exists(TRAIN_OUT) and os.listdir(TRAIN_OUT):
                HF_API.upload_folder(
                    folder_path=TRAIN_OUT,
                    path_in_repo="train",
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                    commit_message="Batch sync: dual-sided neural bracketed train shards"
                )
            if os.path.exists(VAL_OUT) and os.listdir(VAL_OUT):
                HF_API.upload_folder(
                    folder_path=VAL_OUT,
                    path_in_repo="val",
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                    commit_message="Batch sync: dual-sided neural bracketed val shards"
                )
            if os.path.exists(DICT_OUT) and os.listdir(DICT_OUT):
                HF_API.upload_folder(
                    folder_path=DICT_OUT,
                    path_in_repo="dictionaries",
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                    commit_message="Batch sync: multi-candidate dictionaries"
                )
            print(f">> [HF BATCH SYNC] Successfully synced to {self.hf_repo} in {time.time()-t0:.1f}s!\n", flush=True)
        except Exception as e:
            print(f">> [HF BATCH SYNC] Warning: batch sync encountered: {e}\n", flush=True)


def run_neural_align(model_path: str, pairs: List[Tuple[str, str]], batch_size: int = 256, num_workers: int = 8) -> List[str]:
    tmp_dir = tempfile.mkdtemp(prefix="awesome_align_")
    in_file = os.path.join(tmp_dir, "input.txt")
    out_file = os.path.join(tmp_dir, "output.align")

    with open(in_file, "w", encoding="utf-8") as f:
        for en_tok_str, af_tok_str in pairs:
            f.write(f"{en_tok_str} ||| {af_tok_str}\n")

    cmd = [
        ALIGNER_BIN,
        "--model_name_or_path", model_path,
        "--data_file", in_file,
        "--output_file", out_file,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--extraction", "softmax",
        "--align_layer", "8"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    with open(out_file, "r", encoding="utf-8") as f:
        alignments = [line.strip() for line in f]

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return alignments


def process_language(code: str, model_path: str, nlp, batch_size: int = 256):
    info = REGISTRY[code]
    cname = info["clean_name"]
    tag = info["tag"]

    train_dest = os.path.join(TRAIN_OUT, f"{code}.parquet")
    dict_dest = os.path.join(DICT_OUT, f"{code}_dictionary.tsv.gz")
    if os.path.exists(train_dest) and os.path.exists(dict_dest):
        try:
            md = pq.read_metadata(train_dest)
            if md.num_rows > 0:
                return code, md.num_rows, 0, 0, "Already completed"
        except Exception:
            pass

    all_pairs: List[Tuple[str, str, str]] = []
    seen_hashes = set()

    for repo in ["AfriSpeech/africa-corpus", "AfriSpeech/african-corpus-jw", "AfriSpeech/africa-corpus-journals"]:
        try:
            os.environ["AFRICA_CORPUS_REPO"] = repo
            ac._FILE_LIST = None
            ac._REGISTRY = None
            l_obj = ac.resolve(code)
            af_verses = ac._load_verses(l_obj)
            en_verses = ac._load_verses(ac.resolve("en"))
            shared = set(af_verses) & set(en_verses)
            for k in shared:
                for en_t in en_verses[k]:
                    e_clean = clean_txt(en_t)
                    for af_t in af_verses[k]:
                        a_clean = clean_txt(af_t)
                        if len(e_clean) >= 4 and len(a_clean) >= 4:
                            h = f"{e_clean}|||{a_clean}"
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                all_pairs.append((k, e_clean, a_clean))
        except Exception:
            pass

    if not all_pairs:
        return code, 0, 0, 0, "No pairs extracted"

    # 1. Batch populate English phrase cache across all CPU cores
    en_sentences = [e for _, e, _ in all_pairs]
    populate_en_cache(en_sentences, nlp, n_process=8)

    # 2. Build aligned pairs with whitespace tokenization
    input_pairs = []
    parsed_pairs = []
    for vk, en_text, af_text in all_pairs:
        en_toks, phrases, spans = EN_PARSE_CACHE[en_text]
        af_toks = af_text.split()
        input_pairs.append((" ".join(en_toks), " ".join(af_toks)))
        parsed_pairs.append((vk, en_text, af_text, en_toks, af_toks, phrases, spans))

    # 3. GPU Neural Alignment
    align_lines = run_neural_align(model_path, input_pairs, batch_size=batch_size, num_workers=8)

    # 4. Direct linear-time dual-sided projection
    train_rows = []
    val_rows = []
    dict_counts = defaultdict(lambda: defaultdict(int))

    for idx, (vk, en_text, af_text, en_toks, af_toks, phrases, spans) in enumerate(parsed_pairs):
        is_val = is_val_key(vk)
        target_list = val_rows if is_val else train_rows

        if not phrases or not spans:
            target_list.append({
                "instruction": f"Translate the text to {cname}:  {en_text}",
                "response": f"{tag} {af_text}",
                "lang": code,
                "direction": "en2af",
                "verse": vk
            })
            target_list.append({
                "instruction": f"Translate the text to English:  {af_text}",
                "response": f"<eng> {en_text}",
                "lang": code,
                "direction": "af2en",
                "verse": vk
            })
            continue

        raw_align = align_lines[idx] if idx < len(align_lines) else ""
        links = [tuple(map(int, x.split("-"))) for x in raw_align.split() if "-" in x]

        af_rep = {}
        af_rep_rev = {}
        rev_rep = {}

        for s_start, s_end, p in spans:
            aligned_tgts = [j for (ei, j) in links if s_start <= ei < s_end and j < len(af_toks)]
            if aligned_tgts:
                min_t, max_t = min(aligned_tgts), max(aligned_tgts)
                target_phrase = " ".join(af_toks[min_t:max_t+1])
                clean_target = re.sub(r"[^\w\u00C0-\u024F\u1E00-\u1EFF\u0180-\u024FɛɔƐƆ]", "", target_phrase.lower())
                clean_p = p.lower()
                if len(clean_target) >= 2:
                    dict_counts[clean_p][clean_target] += 1
                for tj in range(min_t, max_t + 1):
                    af_rep[tj] = f"<{p}>"
                    af_rep_rev[tj] = f"<{target_phrase}>"
                for ei in range(s_start, s_end):
                    rev_rep[ei] = f"<{target_phrase}>"

        # 1. Forward English with brackets
        en_rep_fwd = {i: f"<{p}>" for s, e, p in spans for i in range(s, e)}
        marked_en_list = []
        skip = -1
        for i, tok in enumerate(en_toks):
            if i <= skip:
                continue
            if i in en_rep_fwd:
                b = en_rep_fwd[i]
                marked_en_list.append(b)
                while i + 1 in en_rep_fwd and en_rep_fwd[i+1] == b:
                    i += 1
                skip = i
            else:
                marked_en_list.append(tok)
        marked_en = " ".join(marked_en_list)

        # 2. Forward African with matching brackets
        marked_af_list = []
        skip = -1
        for j, tok in enumerate(af_toks):
            if j <= skip:
                continue
            if j in af_rep:
                b = af_rep[j]
                marked_af_list.append(b)
                while j + 1 in af_rep and af_rep[j+1] == b:
                    j += 1
                skip = j
            else:
                marked_af_list.append(tok)
        marked_af = " ".join(marked_af_list)

        # 3. Reverse African with matching brackets
        marked_af_rev_list = []
        skip = -1
        for j, tok in enumerate(af_toks):
            if j <= skip:
                continue
            if j in af_rep_rev:
                b = af_rep_rev[j]
                marked_af_rev_list.append(b)
                while j + 1 in af_rep_rev and af_rep_rev[j+1] == b:
                    j += 1
                skip = j
            else:
                marked_af_rev_list.append(tok)
        marked_af_rev = " ".join(marked_af_rev_list)

        # 4. Reverse English with matching brackets
        marked_en_rev_list = []
        skip = -1
        for i, tok in enumerate(en_toks):
            if i <= skip:
                continue
            if i in rev_rep:
                b = rev_rep[i]
                marked_en_rev_list.append(b)
                while i + 1 in rev_rep and rev_rep[i+1] == b:
                    i += 1
                skip = i
            else:
                marked_en_rev_list.append(tok)
        marked_en_rev = " ".join(marked_en_rev_list)

        # Dual-Sided Training Instances
        target_list.append({
            "instruction": f"Translate the text to {cname}:  {marked_en}",
            "response": f"{tag} {marked_af}",
            "lang": code,
            "direction": "en2af",
            "verse": vk
        })
        target_list.append({
            "instruction": f"Translate the text to English:  {marked_af_rev}",
            "response": f"<eng> {marked_en_rev}",
            "lang": code,
            "direction": "af2en",
            "verse": vk
        })

    schema = pa.schema([
        ("instruction", pa.string()),
        ("response", pa.string()),
        ("lang", pa.string()),
        ("direction", pa.string()),
        ("verse", pa.string()),
    ])
    if train_rows:
        pq.write_table(pa.Table.from_pylist(train_rows, schema=schema), train_dest, compression="zstd")
    if val_rows:
        pq.write_table(pa.Table.from_pylist(val_rows, schema=schema), os.path.join(VAL_OUT, f"{code}.parquet"), compression="zstd")

    dict_entries = 0
    with gzip.open(dict_dest, "wt", encoding="utf-8") as gf:
        gf.write(f"# English -> {cname} ({code}) Multi-Candidate Lexicon (Neural Aligned)\n")
        for en_w, af_dict in sorted(dict_counts.items()):
            sorted_cands = sorted(af_dict.items(), key=lambda x: -x[1])
            top_cands = [cw for cw, cnt in sorted_cands if cnt >= 2][:5]
            if top_cands:
                gf.write(f"{en_w}\t" + "\t".join(top_cands) + "\n")
                dict_entries += 1

    return code, len(train_rows), len(val_rows), dict_entries, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to fine-tuned awesome-align checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--langs", default=None, help="Comma-separated list of language codes or None for all 693")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HF dataset repo for live auto-push")
    parser.add_argument("--sync-interval", type=int, default=1800, help="Sync interval in seconds (default: 1800 = 30 mins)")
    args = parser.parse_args()

    codes = sorted(REGISTRY.keys())
    if args.langs:
        codes = [c.strip() for c in args.langs.split(",") if c.strip() in REGISTRY]

    print(f"=== Starting Pan-African Dual-Sided Neural AfriKrane Dataset Generation ===")
    print(f"Model: {args.model}")
    print(f"Total Languages: {len(codes)}")
    print(f"Hugging Face Batch Sync: {args.hf_repo} (Interval: {args.sync_interval}s / {args.sync_interval/60:.0f}m)")

    # Start periodic batch sync daemon
    syncer = PeriodicSyncThread(args.hf_repo, sync_interval_secs=args.sync_interval)
    if args.hf_repo:
        syncer.start()

    nlp = get_nlp()
    t0 = time.time()
    total_train, total_val, total_dicts, success = 0, 0, 0, 0

    for i, code in enumerate(codes, 1):
        c_t0 = time.time()
        c, n_tr, n_val, n_dict, err = process_language(code, args.model, nlp, batch_size=args.batch_size)
        if err:
            print(f"[{i:3d}/{len(codes)}] {code}: {err}", flush=True)
        else:
            total_train += n_tr
            total_val += n_val
            total_dicts += n_dict
            success += 1
            el = time.time() - c_t0
            print(f"[{i:3d}/{len(codes)}] {code} ({REGISTRY[code]['clean_name']}) | Train: {n_tr:,} | Val: {n_val:,} | Dict: {n_dict:,} | {el:.1f}s", flush=True)

    # Final flush to HF
    if args.hf_repo:
        print("\n>> Performing final batch sync to Hugging Face...", flush=True)
        syncer.sync_now()

    tot_time = time.time() - t0
    print(f"\n==========================================")
    print(f"Pan-African Neural Generation Complete in {tot_time/60:.1f}m ({tot_time/3600:.1f}h)!")
    print(f"Successful languages: {success}/{len(codes)}")
    print(f"Total Train Turns: {total_train:,} | Total Val Turns: {total_val:,}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
