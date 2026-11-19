"""AfriKrane: Pan-African Machine Translation with Preserved Noun Phrasing & Contextual Lexical Disambiguation."""

from .registry import LanguageRegistry
from .lookup import PanAfricanLookup
from .aligner import AfroAwesomeAligner
from .extractor import NounPhraseExtractor
from .marker import BracketPostProcessor
from .reranker import ContextualReranker
from .pipeline import AfriKranePipeline

__version__ = "0.1.0"
__all__ = [
    "LanguageRegistry",
    "PanAfricanLookup",
    "AfroAwesomeAligner",
    "NounPhraseExtractor",
    "BracketPostProcessor",
    "ContextualReranker",
    "AfriKranePipeline",
]
