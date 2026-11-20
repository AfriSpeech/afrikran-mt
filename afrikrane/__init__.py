"""AfriKrane: Pan-African Machine Translation with Preserved Noun Phrasing & Contextual Lexical Disambiguation."""

from .registry import LanguageRegistry
from .lookup import PanAfricanLookup
from .aligner import AfroAwesomeAligner
from .extractor import extract_selective_noun_phrases, get_nlp
from .marker import NounPhraseMarker, mark_text_with_phrases, BracketPostProcessor
from .reranker import ContextualReranker
from .pipeline import AfriKranePipeline

__version__ = "0.1.0"
__all__ = [
    "LanguageRegistry",
    "PanAfricanLookup",
    "AfroAwesomeAligner",
    "extract_selective_noun_phrases",
    "get_nlp",
    "NounPhraseMarker",
    "mark_text_with_phrases",
    "BracketPostProcessor",
    "ContextualReranker",
    "AfriKranePipeline",
]
