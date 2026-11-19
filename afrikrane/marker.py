"""Preserved Bracket Parser & Glossary Substitution for Post-Translation."""

import re
from typing import Callable, Dict, List, Optional, Tuple

_BRACKET_PATTERN = re.compile(r"<([^<>\n]+)>")


class BracketPostProcessor:
    """Detects <bracketed> preserved terms in translated text and performs substitution."""

    @staticmethod
    def extract_bracketed_terms(text: str) -> List[str]:
        return _BRACKET_PATTERN.findall(text)

    @staticmethod
    def substitute(
        text: str,
        replacer: Callable[[str], str]
    ) -> str:
        """Replaces each <term> with replacer(term)."""
        def _sub(match):
            term = match.group(1).strip()
            res = replacer(term)
            return res if res else term

        return _BRACKET_PATTERN.sub(_sub, text)

    @staticmethod
    def strip_brackets(text: str) -> str:
        """Removes angle brackets, leaving the internal content intact."""
        return _BRACKET_PATTERN.sub(r"\1", text)
