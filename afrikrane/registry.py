"""Pan-African Language Registry & Metadata Resolver for 693 African Languages."""

import json
import os
from typing import Dict, Optional, Tuple

_DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "african_languages_registry.json"
)

ISO_2_TO_3 = {
    "tw": "twi", "yo": "yor", "ha": "hau", "ig": "ibo", "sw": "swh", "zu": "zul",
    "xh": "xho", "af": "afr", "am": "amh", "so": "som", "om": "orm", "ti": "tir",
    "sn": "sna", "ny": "nya", "st": "sot", "tn": "tsn", "ts": "tso", "ve": "ven",
    "ss": "ssw", "nr": "nbl", "nd": "nde", "sg": "sag", "lg": "lug", "ln": "lin",
    "rw": "kin", "rn": "run", "mg": "mlg", "wo": "wol", "ff": "ful", "ee": "ewe",
    "bm": "bam", "kg": "kon", "lu": "lub", "kj": "kua", "ng": "ndo", "kr": "knc"
}


class LanguageRegistry:
    """Manages metadata, ISO-639-3 codes, special tokens, and names for 693 African languages."""

    def __init__(self, registry_path: Optional[str] = None):
        path = registry_path or _DEFAULT_REGISTRY_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Registry file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.languages: Dict[str, Dict] = data.get("languages", {})
        self.code_to_tag: Dict[str, str] = {k: v["tag"] for k, v in self.languages.items()}
        self.tag_to_code: Dict[str, str] = {v["tag"]: k for k, v in self.languages.items()}
        self.name_to_code: Dict[str, str] = {v["clean_name"].lower(): k for k, v in self.languages.items()}
        self.raw_to_code: Dict[str, str] = {v["raw_name"].lower(): k for k, v in self.languages.items()}

    def resolve_code(self, identifier: str) -> str:
        """Resolves any 2-letter, 3-letter, or language name into a 3-letter ISO code."""
        c = identifier.strip().lower()
        if c in self.languages:
            return c
        if c in ISO_2_TO_3:
            return ISO_2_TO_3[c]
        if c in self.name_to_code:
            return self.name_to_code[c]
        if c in self.raw_to_code:
            return self.raw_to_code[c]
        for name, code in self.name_to_code.items():
            if name in c or c in name:
                return code
        return c

    def get_tag(self, code: str) -> str:
        code_3 = self.resolve_code(code)
        return self.code_to_tag.get(code_3, f"<{code_3}>")

    def get_name(self, code: str) -> str:
        code_3 = self.resolve_code(code)
        info = self.languages.get(code_3)
        return info["clean_name"] if info else code_3.upper()

    def __len__(self):
        return len(self.languages)
