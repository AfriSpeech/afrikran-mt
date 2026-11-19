"""
build_nkrane_dataset_neural.py — Pan-African Neural Alignment & Nkrane Preserved Bracket Generator.

Uses fine-tuned awesome-align (AfriSpeech/afro-awesome-align) on GPU:
1. AfriSpeech/africa-corpus (693 languages, Bible)
2. AfriSpeech/african-corpus-jw (129 languages, JW literature)
3. AfriSpeech/africa-corpus-journals (24 languages, Academic journals)

Projects English noun spans into:
- en -> af: African sentence with <english_noun> in place
- af -> en: English sentence with <african_noun> in place
Exports & auto-pushes:
- data/train_bracketed/{code}.parquet -> AfriSpeech/afrikrane-mt: train/{code}.parquet
- data/val_bracketed/{code}.parquet -> AfriSpeech/afrikrane-mt: val/{code}.parquet
- dictionaries/{code}_dictionary.tsv.gz -> AfriSpeech/afrikrane-mt: dictionaries/{code}_dictionary.tsv.gz
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
import spacy

try:
    from huggingface_hub import HfApi
    HF_API = HfApi()
except ImportError:
    HF_API = None

sys.path.insert(0, "/mnt/volume_d2wey28/projects/africa-corpus-builder")
import africa_corpus as ac

PROJECT_DIR = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_OUT = os.path.join(DATA_DIR, "train_bracketed")
VAL_OUT = os.path.join(DATA_DIR, "val_bracketed")
DICT_OUT = os.path.join(PROJECT_DIR, "dictionaries")

ALIGNER_BIN = "/mnt/volume_d2wey28/projects/afrikrane-mt/.venv/bin/awesome-align"
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "aligner", "model_final")
DEFAULT_HF_REPO = "AfriSpeech/afrikrane-mt"

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(VAL_OUT, exist_ok=True)
os.makedirs(DICT_OUT, exist_ok=True)

REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "african_languages_registry.json"
)
if not os.path.exists(REGISTRY_PATH):
    REGISTRY_PATH = os.path.join(PROJECT_DIR, "african_languages_registry.json")

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

def extract_base_noun_chunks(doc) -> Dict[Tuple[int, int], str]:
    res = {}
    for chunk in doc.noun_chunks:
        root = chunk.root
        if len(root.text.strip()) >= 3 and not root.is_stop and root.pos_ in ("NOUN", "PROPN"):
            words = [t.text for t in chunk if not (t.is_stop and t != root)]
            phrase = " ".join(words).strip()
            if not phrase:
                phrase = root.text.strip()
            res[(chunk.start, chunk.end)] = phrase
    return res

def async_upload_language(code: str, cname: str, hf_repo: str):
    """Background upload of Parquet shards and dictionary to Hugging Face."""
    def _upload():
        if not HF_API or not hf_repo:
            return
        t_path = os.path.join(TRAIN_OUT, f"{code}.parquet")
        v_path = os.path.join(VAL_OUT, f"{code}.parquet")
        d_path = os.path.join(DICT_OUT, f"{code}_dictionary.tsv.gz")
        
        try:
            if os.path.exists(t_path):
                HF_API.upload_file(
                    path_or_fileobj=t_path,
                    path_in_repo=f"train/{code}.parquet",
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"Add {cname} ({code}) neural bracketed train shard"
                )
            if os.path.exists(v_path):
                HF_API.upload_file(
                    path_or_fileobj=v_path,
                    path_in_repo=f"val/{code}.parquet",
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"Add {cname} ({code}) neural bracketed val shard"
                )
            if os.path.exists(d_path):
                HF_API.upload_file(
                    path_or_fileobj=d_path,
                    path_in_repo=f"dictionaries/{code}_dictionary.tsv.gz",
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"Add {cname} ({code}) multi-candidate dictionary"
                )
            print(f">> [HF SYNC] {code} ({cname}) auto-pushed to {hf_repo}")
        except Exception as e:
            print(f"Warning: HF auto-push failed for {code}: {e}")

    threading.Thread(target=_upload, daemon=True).start()

def run_neural_align(model_path: str, pairs: List[Tuple[str, str]], batch_size: int = 256, num_workers: int = 8) -> List[str]:
    tmp_dir = tempfile.mkdtemp(prefix="awesome_align_")
    in_file = os.path.join(tmp_dir, "input.txt")
    out_file = os.path.join(tmp_dir, "output.align")
    
    with open(in_file, "w", encoding="utf-8") as f:
        for en_text, af_text in pairs:
            e_tok = " ".join(re.findall(r"\S+", en_text))
            a_tok = " ".join(re.findall(r"\S+", af_text))
            f.write(f"{e_tok} ||| {a_tok}\n")
            
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

def process_language(code: str, model_path: str, nlp, batch_size: int = 256, hf_repo: Optional[str] = DEFAULT_HF_REPO):
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
        
    input_pairs = [(e, a) for _, e, a in all_pairs]
    align_lines = run_neural_align(model_path, input_pairs, batch_size=batch_size, num_workers=8)
    
    train_rows = []
    val_rows = []
    dict_counts = defaultdict(lambda: defaultdict(int))
    
    for idx, (vk, en_text, af_text) in enumerate(all_pairs):
        is_val = is_val_key(vk)
        target_list = val_rows if is_val else train_rows
        
        doc = nlp(en_text)
        en_tokens = [t.text for t in doc]
        af_tokens = re.findall(r"\S+", af_text)
        
        raw_align = align_lines[idx] if idx < len(align_lines) else ""
        links = [tuple(map(int, x.split("-"))) for x in raw_align.split() if "-" in x]
        noun_spans = extract_base_noun_chunks(doc)
        
        af_rep = {}
        pair_mappings = []
        for (n_start, n_end), noun_str in noun_spans.items():
            aligned_targets = [j for (ei, j) in links if n_start <= ei < n_end and j < len(af_tokens)]
            if aligned_targets:
                min_t, max_t = min(aligned_targets), max(aligned_targets)
                target_phrase = " ".join(af_tokens[min_t:max_t+1])
                clean_target = re.sub(r"[^\w\u00C0-\u024F\u1E00-\u1EFF\u0180-\u024FɛɔƐƆ]", "", target_phrase.lower())
                clean_noun = noun_str.lower()
                if len(clean_target) >= 2:
                    dict_counts[clean_noun][clean_target] += 1
                    pair_mappings.append((noun_str, target_phrase))
                for tj in range(min_t, max_t + 1):
                    af_rep[tj] = f"<{noun_str}>"
                    
        bracketed_af = []
        skip_until = -1
        for tj, tok in enumerate(af_tokens):
            if tj <= skip_until:
                continue
            if tj in af_rep:
                b_str = af_rep[tj]
                bracketed_af.append(b_str)
                while tj + 1 in af_rep and af_rep[tj+1] == b_str:
                    tj += 1
                skip_until = tj
            else:
                bracketed_af.append(tok)
        final_bracketed_af = " ".join(bracketed_af)
        
        bracketed_en = []
        en_rep = {}
        for en_noun, af_noun in pair_mappings:
            for (ns, ne), nstr in noun_spans.items():
                if nstr.lower() == en_noun.lower():
                    for ei in range(ns, ne):
                        en_rep[ei] = f"<{af_noun}>"
        skip_e = -1
        for ei, tok in enumerate(en_tokens):
            if ei <= skip_e:
                continue
            if ei in en_rep:
                b_str = en_rep[ei]
                bracketed_en.append(b_str)
                while ei + 1 in en_rep and en_rep[ei+1] == b_str:
                    ei += 1
                skip_e = ei
            else:
                bracketed_en.append(tok)
        final_bracketed_en = " ".join(bracketed_en)
        
        target_list.append({
            "instruction": f"Translate the text to {cname}:  {en_text}",
            "response": f"{tag} {final_bracketed_af}",
            "lang": code,
            "direction": "en2af",
            "verse": vk
        })
        target_list.append({
            "instruction": f"Translate the text to English:  {af_text}",
            "response": f"<eng> {final_bracketed_en}",
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
                
    if hf_repo:
        async_upload_language(code, cname, hf_repo)
        
    return code, len(train_rows), len(val_rows), dict_entries, None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to fine-tuned awesome-align checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--langs", default=None, help="Comma-separated list of language codes or None for all 693")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HF dataset repo for live auto-push")
    args = parser.parse_args()
    
    codes = sorted(REGISTRY.keys())
    if args.langs:
        codes = [c.strip() for c in args.langs.split(",") if c.strip() in REGISTRY]
        
    print(f"=== Starting Pan-African Neural AfriKrane Dataset Generation ===")
    print(f"Model: {args.model}")
    print(f"Total Languages: {len(codes)}")
    print(f"Hugging Face Live Sync: {args.hf_repo}")
    
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    t0 = time.time()
    total_train, total_val, total_dicts, success = 0, 0, 0, 0
    
    for i, code in enumerate(codes, 1):
        c_t0 = time.time()
        c, n_tr, n_val, n_dict, err = process_language(code, args.model, nlp, batch_size=args.batch_size, hf_repo=args.hf_repo)
        if err:
            print(f"[{i:3d}/{len(codes)}] {code}: {err}")
        else:
            total_train += n_tr
            total_val += n_val
            total_dicts += n_dict
            success += 1
            el = time.time() - c_t0
            print(f"[{i:3d}/{len(codes)}] {code} ({REGISTRY[code]['clean_name']}) | Train: {n_tr:,} | Val: {n_val:,} | Dict: {n_dict:,} | {el:.1f}s")
            
    tot_time = time.time() - t0
    print(f"\n==========================================")
    print(f"Pan-African Neural Generation Complete in {tot_time/60:.1f}m ({tot_time/3600:.1f}h)!")
    print(f"Successful languages: {success}/{len(codes)}")
    print(f"Total Train Turns: {total_train:,} | Total Val Turns: {total_val:,}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
