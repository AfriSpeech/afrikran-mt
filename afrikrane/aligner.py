"""Afro-Awesome-Align Neural Wrapper for Cross-Lingual Phrase & Word Alignment."""

import re
import subprocess
import tempfile
import os
import shutil
from typing import List, Tuple, Optional

DEFAULT_MODEL = "AfriSpeech/afro-awesome-align"


class AfroAwesomeAligner:
    """Pan-African neural word & phrase aligner wrapper."""

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, device: str = "cuda:0", batch_size: int = 128):
        self.model_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size

    def align_pairs(self, pairs: List[Tuple[str, str]], align_layer: int = 8) -> List[List[Tuple[int, int]]]:
        """Runs fast batched alignment using awesome-align extraction."""
        tmp_dir = tempfile.mkdtemp(prefix="afro_align_")
        in_path = os.path.join(tmp_dir, "input.txt")
        out_path = os.path.join(tmp_dir, "output.align")

        with open(in_path, "w", encoding="utf-8") as f:
            for src, tgt in pairs:
                s_tok = " ".join(re.findall(r"\S+", src))
                t_tok = " ".join(re.findall(r"\S+", tgt))
                f.write(f"{s_tok} ||| {t_tok}\n")

        cmd = [
            "awesome-align",
            "--model_name_or_path", self.model_path,
            "--data_file", in_path,
            "--output_file", out_path,
            "--batch_size", str(self.batch_size),
            "--extraction", "softmax",
            "--align_layer", str(align_layer),
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            with open(out_path, "r", encoding="utf-8") as f:
                raw_lines = [l.strip() for l in f]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        results = []
        for line in raw_lines:
            links = []
            for item in line.split():
                if "-" in item:
                    try:
                        i, j = item.split("-")
                        links.append((int(i), int(j)))
                    except ValueError:
                        pass
            results.append(links)
        return results
