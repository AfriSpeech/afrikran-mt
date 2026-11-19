"""
inference.py — Pan-African translation engine supporting 693 African languages.
"""

import argparse
import json
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(PROJECT_DIR, "african_languages_registry.json")

with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    REG_DATA = json.load(f)["languages"]

# Name/Code mappings
CODE_TO_NAME = {k: v["clean_name"] for k, v in REG_DATA.items()}
NAME_TO_CODE = {v["clean_name"].lower(): k for k, v in REG_DATA.items()}
RAW_TO_CODE = {v["raw_name"].lower(): k for k, v in REG_DATA.items()}

def resolve_target(target: str):
    t_low = target.strip().lower()
    if t_low in ("en", "eng", "english"):
        return "English", "<eng>"
    if t_low in CODE_TO_NAME:
        return CODE_TO_NAME[t_low], f"<{t_low}>"
    if t_low in NAME_TO_CODE:
        code = NAME_TO_CODE[t_low]
        return CODE_TO_NAME[code], f"<{code}>"
    if t_low in RAW_TO_CODE:
        code = RAW_TO_CODE[t_low]
        return CODE_TO_NAME[code], f"<{code}>"
    # Fallback to title case
    return target.strip(), f"<{t_low[:3]}>"

class NkraneAfricaTranslator:
    def __init__(self, model_path_or_repo: str = None, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        path = model_path_or_repo or os.path.join(PROJECT_DIR, "checkpoints", "best")
        if not os.path.exists(path):
            path = os.path.join(PROJECT_DIR, "tokenizer")
        
        print(f"Loading Nkrane African MT engine from: {path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_weights_path = model_path_or_repo or os.path.join(PROJECT_DIR, "checkpoints", "best")
        if not os.path.exists(model_weights_path):
            model_weights_path = "AfriSpeech/gemma-3-1b-translate-sft-best"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_weights_path,
            dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            device_map=self.device,
        )
        if self.model.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def translate(self, text: str, target: str, max_new_tokens: int = 128) -> str:
        tgt_name, tag = resolve_target(target)
        if tgt_name == "English":
            prompt = f"<bos><start_of_turn>user\nTranslate the text to English:  {text.strip()}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"<bos><start_of_turn>user\nTranslate the text to {tgt_name}:  {text.strip()}<end_of_turn>\n<start_of_turn>model\n"

        enc = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            )
        raw = self.tokenizer.decode(gen[0, enc.shape[1]:], skip_special_tokens=True).strip()
        # Clean prefix tag (e.g. "<yor> ...")
        clean = re.sub(r"^<[a-z]{2,5}>\s*", "", raw)
        return clean.strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Path to checkpoint")
    parser.add_argument("--text", default=None, help="Text to translate")
    parser.add_argument("--target", default="Twi", help="Target language (code or name)")
    parser.add_argument("--interactive", action="store_true", help="Interactive CLI")
    args = parser.parse_args()

    engine = NkraneAfricaTranslator(model_path_or_repo=args.model)

    if args.text:
        out = engine.translate(args.text, args.target)
        print(f"\nTarget: {args.target}")
        print(f"Translation: {out}\n")
    elif args.interactive:
        print("\nEntering interactive mode. Type quit to exit.\n")
        tgt = args.target
        while True:
            try:
                line = input(f"[{tgt}] > ")
                if line.strip().lower() in ("quit", "exit"):
                    break
                if line.startswith(":lang "):
                    tgt = line.split(" ", 1)[1].strip()
                    print(f"Switched target language to: {tgt}")
                    continue
                res = engine.translate(line, tgt)
                print(f"=> {res}\n")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
