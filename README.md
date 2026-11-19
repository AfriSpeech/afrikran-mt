# AfriKrane-MT: Pan-African Machine Translation across 693 Languages

**AfriKrane-MT** is a Pan-African neural machine translation framework designed to support **693 African languages** with **preserved noun phrasing** and **contextual lexical disambiguation** (Nkrane-MT architecture extended across the African continent).

---

## Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │    English Source Text                │
                      └──────────────────┬────────────────────┘
                                         │
                   ┌─────────────────────▼─────────────────────┐
                   │    AfriSpeech/nkrane-gemma-3-1b-all-africa │  (Generative LLM)
                   │    Preserved-Bracket Generation           │  Produces <noun> spans
                   └─────────────────────┬─────────────────────┘
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               │                                                   │
               ▼                                                   ▼
┌──────────────────────────────┐              ┌───────────────────────────────────────┐
│ User Custom Glossary Lookup   │              │ Multi-Candidate Lexicon (693 langs)   │
│ (Zero-shot deterministic sub)│              │ Extracted via Afro-Awesome-Align      │
└──────────────┬───────────────┘              └───────────────────┬───────────────────┘
               │                                                   │
               └─────────────────────────┬─────────────────────────┘
                                         │
                   ┌─────────────────────▼─────────────────────┐
                   │    Contextual Lexical Reranker            │  (Afro-XLMR Cross-Encoder)
                   │    Selects Contextually Optimal Synonym   │
                   └─────────────────────┬─────────────────────┘
                                         │
                      ┌──────────────────▼────────────────────┐
                      │    Final Fluent African Target Text   │
                      └───────────────────────────────────────┘
```

---

## Key Components

1. **Afro-Awesome-Align ([`AfriSpeech/afro-awesome-align`](https://huggingface.co/AfriSpeech/afro-awesome-align)):**
   - Fine-tuned across 329,175 parallel sentence pairs spanning all 693 African languages on an NVIDIA H200 GPU.
   - Provides neural cross-attention representations for token and multi-word phrase span projection.

2. **Pan-African Gemma-3-1B ([`AfriSpeech/nkrane-gemma-3-1b-all-africa`](https://huggingface.co/AfriSpeech/nkrane-gemma-3-1b-all-africa)):**
   - Expanded vocabulary: 601 Pan-African language tags (`<twi>`, `<yor>`, `<hau>`, `<swh>`, `<zul>`, `<eng>`, etc.). Total vocabulary: 262,746.
   - Fine-tuned on 110M+ parallel turns across Africa Corpus Builder sources (Bible, JW, Journals).
   - Generates translations while preserving base noun phrases in `<brackets>` for downstream lexical control.

3. **Multi-Candidate Dictionaries (693 Languages):**
   - Automatically extracted from converged neural alignments across all 693 languages.
   - Enables domain-specific and dialectal synonym disambiguation.

4. **Afro-XLMR Contextual Reranker:**
   - Cross-encoder architecture (`Davlan/afro-xlmr-base`) trained on parallel contexts to pick the exact synonym that fits the grammatical and semantic context.

---

## Repository Structure

```
afrikrane-mt/
├── afrikrane/                        # Core Python library
│   ├── registry.py                   # 693-language metadata & ISO resolver
│   ├── lookup.py                     # Bilingual multi-candidate lexicons
│   ├── aligner.py                    # Afro-Awesome-Align wrapper
│   ├── extractor.py                  # Multi-word noun phrase projector
│   ├── marker.py                     # Preserved bracket substitution & strip
│   ├── reranker.py                   # Afro-XLMR cross-encoder reranker
│   └── pipeline.py                   # Full end-to-end translation pipeline
├── data/
│   └── african_languages_registry.json # Complete metadata for 693 languages
├── data_prep/                        # Dataset building pipelines
│   ├── build_registry_and_tokenizer.py # Tokenizer expansion (601 tags)
│   ├── prepare_dataset.py            # Parallel corpus builder (110M turns)
│   ├── prepare_aligner_data.py       # Balanced parallel sampling
│   └── build_nkrane_dataset_neural.py# GPU neural alignment & bracket generator
├── training/                         # Model training recipes
│   ├── run_awesome_train.sh          # Afro-Awesome-Align training recipe
│   ├── evaluate_checkpoints.py       # African benchmark evaluation harness
│   ├── train_gemma.py                # 5-epoch Gemma-3-1B SFT script
│   ├── run_training.sh               # Gemma-3-1B runner script
│   ├── build_reranker_dataset.py     # Pan-African reranker dataset builder
│   └── train_reranker.py             # Afro-XLMR cross-encoder training
└── scripts/
    └── inference.py                  # Live CLI translation & dictionary substitution
```

---

## Quickstart

### Installation
```bash
git clone https://github.com/GhanaOpenAI/afrikrane-mt.git
cd afrikrane-mt
pip install -e .
python -m spacy download en_core_web_sm
```

### Python Translation
```python
from afrikrane import AfriKranePipeline

translator = AfriKranePipeline(
    generator_model="AfriSpeech/nkrane-gemma-3-1b-all-africa",
    device="cuda:0"
)

# Translate English to Asante Twi
output = translator.translate(
    text="The holy spirit gives peace to all people.",
    target_lang="twi"
)
print("Twi:", output)

# Translate English to Swahili with Custom Glossary Override
output_swh = translator.translate(
    text="The teacher provides good news to the community.",
    target_lang="swh",
    custom_glossary={"teacher": "mwalimu mkuu"}
)
print("Swahili:", output_swh)
```

---

## Citation
```bibtex
@software{afrikrane2026,
  author = {Owusu, Michael Seth and Ghana NLP and AfriSpeech},
  title = {AfriKrane-MT: Pan-African Machine Translation across 693 Languages},
  year = {2026},
  url = {https://github.com/GhanaOpenAI/afrikrane-mt}
}
```
Developed with ❤️ by **Ghana NLP** and **AfriSpeech**.
EOF
