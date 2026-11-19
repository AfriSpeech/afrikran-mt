"""Noun Chunk Extractor & Multi-Word Span Projector for Preserved Brackets."""

import re
from typing import Dict, List, Optional, Set, Tuple

try:
    import spacy
except ImportError:
    spacy = None


class NounPhraseExtractor:
    """Extracts base noun phrases from English sentences for preserved bracket translation."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        if spacy is None:
            raise ImportError("Please install spacy: pip install spacy && python -m spacy download en_core_web_sm")
        self.nlp = spacy.load(spacy_model, disable=["ner"])

    def extract_chunks(self, text: str) -> List[Tuple[int, int, str]]:
        """Returns list of (start_token_idx, end_token_idx, clean_noun_text)."""
        doc = self.nlp(text)
        results = []
        for chunk in doc.noun_chunks:
            root = chunk.root
            if len(root.text.strip()) >= 3 and not root.is_stop and root.pos_ in ("NOUN", "PROPN"):
                words = [t.text for t in chunk if not (t.is_stop and t != root)]
                phrase = " ".join(words).strip()
                if not phrase:
                    phrase = root.text.strip()
                results.append((chunk.start, chunk.end, phrase))
        return results

    def project_brackets(
        self,
        en_text: str,
        af_text: str,
        links: List[Tuple[int, int]]
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        """Projects English noun spans into target African text in <noun> brackets."""
        doc = self.nlp(en_text)
        en_tokens = [t.text for t in doc]
        af_tokens = re.findall(r"\S+", af_text)
        chunks = self.extract_chunks(en_text)

        af_rep = {}
        pair_mappings = []

        for c_start, c_end, phrase in chunks:
            aligned_tgts = [j for (ei, j) in links if c_start <= ei < c_end and j < len(af_tokens)]
            if aligned_tgts:
                min_t, max_t = min(aligned_tgts), max(aligned_tgts)
                target_phrase = " ".join(af_tokens[min_t:max_t+1])
                pair_mappings.append((phrase, target_phrase))
                for tj in range(min_t, max_t + 1):
                    af_rep[tj] = f"<{phrase}>"

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

        final_af = " ".join(bracketed_af)

        # Reverse projection
        en_rep = {}
        for en_noun, af_noun in pair_mappings:
            for c_start, c_end, phrase in chunks:
                if phrase.lower() == en_noun.lower():
                    for ei in range(c_start, c_end):
                        en_rep[ei] = f"<{af_noun}>"

        bracketed_en = []
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

        final_en = " ".join(bracketed_en)
        return final_af, final_en, pair_mappings
