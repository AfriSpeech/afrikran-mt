"""Pan-African Multi-Candidate Lexicon Loader & Bilingual Lookup."""

import gzip
import os
import re
from typing import Dict, List, Optional

_WS = re.compile(r"\s+")


def normalize_term(term: str) -> str:
    return _WS.sub(" ", term).strip().lower()


class PanAfricanLookup:
    """Loads and queries multi-candidate bilingual dictionaries across 693 African languages."""

    def __init__(self, dict_dir: Optional[str] = None):
        self.dict_dir = dict_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dictionaries"
        )
        self.loaded_dicts: Dict[str, Dict[str, List[str]]] = {}

    def load_language(self, lang_code: str) -> Dict[str, List[str]]:
        lang_code = lang_code.lower().strip()
        if lang_code in self.loaded_dicts:
            return self.loaded_dicts[lang_code]

        candidates_map: Dict[str, List[str]] = {}
        file_path = os.path.join(self.dict_dir, f"{lang_code}_dictionary.tsv.gz")

        if os.path.exists(file_path):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        en_word = normalize_term(parts[0])
                        cands = [c.strip() for c in parts[1:] if c.strip()]
                        if en_word and cands:
                            candidates_map[en_word] = cands

        self.loaded_dicts[lang_code] = candidates_map
        return candidates_map

    def get_candidates(self, en_noun: str, lang_code: str) -> List[str]:
        d = self.load_language(lang_code)
        return d.get(normalize_term(en_noun), [])

    def translate_noun(self, en_noun: str, lang_code: str) -> Optional[str]:
        cands = self.get_candidates(en_noun, lang_code)
        return cands[0] if cands else None
