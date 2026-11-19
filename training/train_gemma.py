"""
train.py — High-Throughput Bidirectional SFT for Gemma-3-1B across 693 African Languages.

Target: 5 Full Epochs with Early Stopping if no improvement for 0.25 epoch (1 quarter epoch).
Optimized for NVIDIA H200 (Hopper, 140GB VRAM, bfloat16 + TF32).
"""

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import time
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Enable Hopper TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class BufferedAfricanDataset:
    """Rotating in-memory buffer across 693 language Parquet shards."""

    def __init__(self, shard_files: List[str], pool_size: int = 60, samples_per_shard: int = 400):
        self.shard_files = shard_files
        self.pool_size = min(pool_size, len(shard_files))
        self.samples_per_shard = samples_per_shard
        self.buffer: List[Tuple[str, str, str]] = []
        self._refill()

    def _refill(self):
        picked = random.sample(self.shard_files, self.pool_size)
        new_items = []
        for sf in picked:
            try:
                t = pq.read_table(sf, columns=["instruction", "response", "lang"])
                n = len(t)
                k = min(self.samples_per_shard, n)
                indices = random.sample(range(n), k)
                insts = t["instruction"]
                resps = t["response"]
                langs = t["lang"]
                for idx in indices:
                    new_items.append((insts[idx].as_py(), resps[idx].as_py(), langs[idx].as_py()))
            except Exception as e:
                print(f"Warning: could not read {sf}: {e}")
        random.shuffle(new_items)
        self.buffer = new_items

    def get_batch(self, batch_size: int = 32) -> List[Tuple[str, str, str]]:
        if len(self.buffer) < batch_size:
            self._refill()
        batch = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]
        return batch


def collate_and_mask(
    batch: List[Tuple[str, str, str]],
    tok: AutoTokenizer,
    max_seq_len: int = 256,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepares input_ids, attention_mask, and labels (prompt tokens masked with -100)."""
    all_input_ids, all_masks, all_labels = [], [], []

    for inst, resp, _ in batch:
        p_str = f"<bos><start_of_turn>user\n{inst}<end_of_turn>\n<start_of_turn>model\n"
        f_str = f"<bos><start_of_turn>user\n{inst}<end_of_turn>\n<start_of_turn>model\n{resp}<end_of_turn>"

        p_ids = tok.encode(p_str, add_special_tokens=False)
        f_ids = tok.encode(f_str, add_special_tokens=False)[:max_seq_len]

        p_len = min(len(p_ids), len(f_ids))
        lbls = [-100] * p_len + f_ids[p_len:]
        mask = [1] * len(f_ids)

        all_input_ids.append(f_ids)
        all_masks.append(mask)
        all_labels.append(lbls)

    max_len = max(len(x) for x in all_input_ids)
    padded_inputs = [x + [tok.pad_token_id] * (max_len - len(x)) for x in all_input_ids]
    padded_masks = [x + [0] * (max_len - len(x)) for x in all_masks]
    padded_labels = [x + [-100] * (max_len - len(x)) for x in all_labels]

    return (
        torch.tensor(padded_inputs, dtype=torch.long, device=device),
        torch.tensor(padded_masks, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
    )


def evaluate(model, val_dataset: BufferedAfricanDataset, tok, max_seq_len=256, eval_batches=40, device="cuda:0"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_batches):
            batch = val_dataset.get_batch(32)
            input_ids, attention_mask, labels = collate_and_mask(batch, tok, max_seq_len, device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    model.train()
    return total_loss / eval_batches


def sample_generate(model, tok, samples: List[Tuple[str, str]], max_new_tokens=64, device="cuda:0"):
    model.eval()
    print("\n--- Live Translation Inspection ---")
    with torch.no_grad():
        for inst, expected in samples:
            prompt = f"<bos><start_of_turn>user\n{inst}<end_of_turn>\n<start_of_turn>model\n"
            enc = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
            gen = model.generate(
                enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.convert_tokens_to_ids("<end_of_turn>"),
            )
            raw = tok.decode(gen[0, enc.shape[1]:], skip_special_tokens=True).strip()
            print(f"P: {inst[:65]}...")
            print(f"O: {raw[:70]}")
            print(f"R: {expected[:70]}\n")
    print("-----------------------------------\n")
    model.train()


def check_disk_space(path: str, min_free_gb: float = 20.0):
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(f"Disk space low: {free_gb:.1f} GB free < {min_free_gb} GB threshold.")




def push_to_hf_async(ckpt_dir: str, repo_id: str, commit_msg: str):
    import threading
    def _upload():
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                folder_path=ckpt_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg,
            )
            print(f">> Successfully pushed checkpoint to Hugging Face ({repo_id}): {commit_msg}")
        except Exception as e:
            print(f"Warning: HF push failed: {e}")
    threading.Thread(target=_upload, daemon=True).start()

def prune_checkpoints(pattern: str, max_keep: int = 5, key_fn=None):
    dirs = glob.glob(pattern)
    if len(dirs) > max_keep:
        if key_fn is None:
            def default_key(p):
                m = re.findall(r"\d+", os.path.basename(p))
                return int(m[-1]) if m else os.path.getmtime(p)
            key_fn = default_key
        sorted_dirs = sorted(dirs, key=key_fn)
        to_remove = sorted_dirs[:-max_keep]
        for d in to_remove:
            try:
                shutil.rmtree(d)
                print(f">> Pruned older checkpoint: {d}")
            except Exception as e:
                print(f"Warning: could not prune {d}: {e}")

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.05):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="AfriSpeech/gemma-3-1b-translate-sft-best")
    parser.add_argument("--tokenizer", default="/mnt/volume_d2wey28/projects/nkrane-africa-gemma/tokenizer")
    parser.add_argument("--train-dir", default="/mnt/volume_d2wey28/projects/nkrane-africa-gemma/data/train")
    parser.add_argument("--val-dir", default="/mnt/volume_d2wey28/projects/nkrane-africa-gemma/data/val")
    parser.add_argument("--output-dir", default="/mnt/volume_d2wey28/projects/nkrane-africa-gemma/checkpoints")
    parser.add_argument("--hf-repo", default="AfriSpeech/nkrane-gemma-3-1b-all-africa", help="HF repo id to push checkpoints")
    parser.add_argument("--resume", default=None, help="Resume training from checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--epochs", type=float, default=5.0, help="Target number of full epochs")
    parser.add_argument("--epoch-examples", type=int, default=32000000, help="Number of examples per epoch")
    parser.add_argument("--patience-ratio", type=float, default=0.25, help="Early stopping patience as fraction of epoch (0.25 = 1/4 epoch)")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = int(args.epoch_examples / effective_batch)
    max_steps = int(args.epochs * steps_per_epoch)
    patience_steps = int(args.patience_ratio * steps_per_epoch)

    print(f"Loading expanded tokenizer from: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Check for resume checkpoint
    load_path = args.resume or args.base_model
    start_step = 0
    best_val_loss = float("inf")
    steps_without_improvement = 0

    if args.resume and os.path.exists(args.resume):
        print(f">> Resuming model weights from: {args.resume}")
        eval_info_path = os.path.join(args.resume, "eval_info.json")
        if os.path.exists(eval_info_path):
            with open(eval_info_path, "r") as f:
                info = json.load(f)
                start_step = info.get("step", 0)
                best_val_loss = info.get("best_val_loss", float("inf"))
                print(f"Loaded resume state: step={start_step}, best_val_loss={best_val_loss:.4f}")

    print(f"Loading model weights from: {load_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        dtype=torch.bfloat16,
        device_map=device,
    )
    orig_vocab = model.get_input_embeddings().weight.shape[0]
    if orig_vocab != len(tok):
        print(f"Resizing embeddings from {orig_vocab} to {len(tok)}...")
        model.resize_token_embeddings(len(tok))

    model.train()

    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.parquet")))
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*.parquet")))
    print(f"Found {len(train_files)} train shards and {len(val_files)} val shards.")

    train_ds = BufferedAfricanDataset(train_files, pool_size=60, samples_per_shard=400)
    val_ds = BufferedAfricanDataset(val_files, pool_size=40, samples_per_shard=200)

    # Fixed sample checks across major languages
    test_samples = [
        ("Translate the text to Twi:  God so loved the world that he gave his only Son.", "<twi> Na sɛnea Onyankopɔn dɔɔ wiase ni"),
        ("Translate the text to Yoruba:  In the beginning was the Word, and the Word was with God.", "<yor> Ní ìbẹ̀rẹ̀ ni Ọ̀rọ̀ wà, Ọ̀rọ̀ sì wà pẹ̀lú Ọlọ́run"),
        ("Translate the text to Hausa:  The light shines in the darkness, and the darkness has not overcome it.", "<hau> Haske yana haskakawa a cikin duhu"),
        ("Translate the text to Swahili:  The true light that gives light to everyone was coming into the world.", "<swh> Nuru ya kweli imwangaziayo kila mtu"),
        ("Translate the text to Dagbani:  He was in the world, and though the world was made through him.", "<dag> O daa bemi tiŋgbani maa ni"),
        ("Translate the text to English:  Mfitiaseɛ no Onyankopɔn bɔɔ ɔsoro ne asase.", "<eng> In the beginning God created the heavens and the earth."),
    ]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(opt, args.warmup_steps, max_steps)

    # Fast forward scheduler to current step if resuming
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    print("\n==========================================")
    print(f"Training started on {device} ({torch.cuda.get_device_name(0)})")
    print(f"Target Epochs: {args.epochs:.1f} ({max_steps:,} total steps)")
    print(f"Steps per Epoch: {steps_per_epoch:,} (Effective Batch Size: {effective_batch})")
    print(f"Early Stopping Patience: {patience_steps:,} steps (0.25 epoch)")
    print(f"Resuming from Step: {start_step:,} | Best Val Loss: {best_val_loss:.4f}")
    print(f"Evaluation every: {args.eval_every} steps | Save every: {args.save_every} steps")
    print("==========================================\n")

    step = start_step
    accum_loss = 0.0
    t_step = time.time()
    tokens_processed = 0

    opt.zero_grad()

    while step < max_steps:
        # Micro-batches for gradient accumulation
        for _ in range(args.grad_accum):
            batch = train_ds.get_batch(args.batch_size)
            input_ids, attention_mask, labels = collate_and_mask(batch, tok, args.max_seq_len, device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            accum_loss += loss.item() * args.grad_accum
            tokens_processed += (labels != -100).sum().item()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        opt.zero_grad()
        step += 1

        # Logging
        if step % 50 == 0 or step == start_step + 1:
            dt = time.time() - t_step
            tok_s = tokens_processed / max(dt, 0.001)
            lr_cur = scheduler.get_last_lr()[0]
            current_epoch = step / steps_per_epoch
            print(
                f"Step {step:7d}/{max_steps} (Epoch {current_epoch:.2f}/{args.epochs:.1f}) | "
                f"Loss: {accum_loss / 50 if step > start_step + 1 else accum_loss:.4f} | "
                f"LR: {lr_cur:.2e} | "
                f"Speed: {tok_s:5.0f} tok/s | "
                f"VRAM: {torch.cuda.max_memory_allocated(device)/1e9:.1f}GB"
            )
            accum_loss = 0.0
            tokens_processed = 0
            t_step = time.time()

        # Disk space check
        if step % 200 == 0:
            check_disk_space(args.output_dir, min_free_gb=15.0)

        # Evaluation & live sample inspection
        if step % args.eval_every == 0:
            val_loss = evaluate(model, val_ds, tok, args.max_seq_len, eval_batches=40, device=device)
            current_epoch = step / steps_per_epoch
            print(f"\n[Evaluation @ Step {step} (Epoch {current_epoch:.2f})] Validation Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")
            sample_generate(model, tok, test_samples, max_new_tokens=64, device=device)

            if val_loss < best_val_loss:
                delta = best_val_loss - val_loss
                best_val_loss = val_loss
                steps_without_improvement = 0
                # 1. Update best symlink/pointer directory
                best_dir = os.path.join(args.output_dir, "best")
                print(f">> Improvement (-{delta:.4f})! Saving best checkpoint (Loss: {val_loss:.4f}) to {best_dir}...")
                model.save_pretrained(best_dir)
                tok.save_pretrained(best_dir)
                with open(os.path.join(best_dir, "eval_info.json"), "w") as f:
                    json.dump({"step": step, "epoch": current_epoch, "val_loss": val_loss, "best_val_loss": best_val_loss}, f, indent=2)
                
                # 2. Save versioned best checkpoint: best_step_{step}
                v_best_dir = os.path.join(args.output_dir, f"best_step_{step}")
                model.save_pretrained(v_best_dir)
                tok.save_pretrained(v_best_dir)
                with open(os.path.join(v_best_dir, "eval_info.json"), "w") as f:
                    json.dump({"step": step, "epoch": current_epoch, "val_loss": val_loss, "best_val_loss": best_val_loss}, f, indent=2)
                
                # 3. Keep strictly the last 5 best checkpoints
                prune_checkpoints(os.path.join(args.output_dir, "best_step_*"), max_keep=5)
                
                # 4. Asynchronously push new best checkpoint to Hugging Face
                if args.hf_repo:
                    push_to_hf_async(best_dir, args.hf_repo, f"New Best Checkpoint @ Step {step} (Val Loss: {val_loss:.4f})")
            else:
                steps_without_improvement += args.eval_every
                epochs_stagnant = steps_without_improvement / steps_per_epoch
                print(f"-- No improvement for {steps_without_improvement:,} steps ({epochs_stagnant:.3f} epochs). Patience threshold: {patience_steps:,} steps (0.25 epoch)")

                if steps_without_improvement >= patience_steps:
                    print(f"\n==========================================")
                    print(f"EARLY STOPPING TRIGGERED @ Step {step} (Epoch {current_epoch:.2f})!")
                    print(f"Validation loss did not improve for {patience_steps:,} steps (0.25 epoch).")
                    print(f"Best Validation Loss: {best_val_loss:.4f}")
                    print(f"==========================================\n")
                    break

        # Periodic latest checkpoint (Keep last 5 latest checkpoints)
        if step % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
            print(f">> Saving periodic checkpoint to {ckpt_dir}...")
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "eval_info.json"), "w") as f:
                json.dump({"step": step, "epoch": current_epoch, "val_loss": val_loss if "val_loss" in locals() else None, "best_val_loss": best_val_loss}, f, indent=2)
            
            # Keep strictly the last 5 latest step checkpoints
            prune_checkpoints(os.path.join(args.output_dir, "step_*"), max_keep=5)

    print(f"\nTraining completed! Final step reached: {step}/{max_steps}")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
