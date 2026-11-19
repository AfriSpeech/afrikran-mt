import json
import re
import os
import sys

sys.path.insert(0, "/mnt/volume_d2wey28/projects/africa-corpus-builder")
import africa_corpus as ac
from transformers import AutoTokenizer

base_model = "AfriSpeech/gemma-3-1b-translate-sft-best"
out_dir = "/mnt/volume_d2wey28/projects/nkrane-africa-gemma"
tok = AutoTokenizer.from_pretrained(base_model)
vocab = tok.get_vocab()

a_langs, _ = ac.list_languages()
print("Total African languages from builder:", len(a_langs))

registry = {}
tokens_to_add = []

if "<eng>" not in vocab:
    tokens_to_add.append("<eng>")

def clean_name(n):
    if "/" in n:
        n = n.split("/")[0].strip()
    n = re.sub(r"\s*\([^)]*\)", "", n).strip()
    n = n.rstrip(".:,;")
    return n

for l in a_langs:
    code = l.code.lower().strip()
    cname = clean_name(l.name)
    tag = f"<{code}>"
    in_base = (tag in vocab)
    
    registry[code] = {
        "code": code,
        "raw_name": l.name,
        "clean_name": cname,
        "tag": tag,
        "files": l.files,
        "in_base_vocab": in_base
    }
    
    if not in_base and tag not in tokens_to_add:
        tokens_to_add.append(tag)

already_count = sum(1 for v in registry.values() if v.get("in_base_vocab"))
print("Tokens already in vocab:", already_count)
print("New tokens to add to tokenizer:", len(tokens_to_add))

added_count = tok.add_tokens(tokens_to_add)
print(f"Successfully added {added_count} tokens to tokenizer!")
print(f"New tokenizer vocab size: {len(tok)}")

tok_dir = os.path.join(out_dir, "tokenizer")
tok.save_pretrained(tok_dir)
print(f"Saved expanded tokenizer to {tok_dir}")

reg_file = os.path.join(out_dir, "african_languages_registry.json")
with open(reg_file, "w", encoding="utf-8") as f:
    json.dump({
        "num_languages": len(registry),
        "tokens_added": len(tokens_to_add),
        "languages": registry
    }, f, indent=2, ensure_ascii=False)
print(f"Saved languages registry to {reg_file}")
