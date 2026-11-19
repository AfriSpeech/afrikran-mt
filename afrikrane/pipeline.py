"""AfriKrane End-to-End Pan-African Preserved Bracket Machine Translation Pipeline."""

from typing import Dict, List, Optional
from .registry import LanguageRegistry
from .lookup import PanAfricanLookup
from .marker import BracketPostProcessor
from .reranker import ContextualReranker


class AfriKranePipeline:
    """Full Pan-African MT with preserved source nouns and contextual lexical disambiguation."""

    def __init__(
        self,
        generator_model: str = "AfriSpeech/nkrane-gemma-3-1b-all-africa",
        reranker_model: Optional[str] = "Davlan/afro-xlmr-base",
        registry_path: Optional[str] = None,
        dict_dir: Optional[str] = None,
        device: str = "cuda:0",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.registry = LanguageRegistry(registry_path)
        self.lookup = PanAfricanLookup(dict_dir)

        print(f"Loading AfriKrane generator: {generator_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            generator_model,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
        ).eval()

        self.reranker = None
        if reranker_model:
            print(f"Loading AfriKrane reranker: {reranker_model}...")
            self.reranker = ContextualReranker(reranker_model, device=self.device)

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "en",
        custom_glossary: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 128,
    ) -> str:
        tgt_code = self.registry.resolve_code(target_lang)
        tgt_name = self.registry.get_name(tgt_code)
        tgt_tag = self.registry.get_tag(tgt_code)

        # 1. Format generation prompt
        if source_lang == "en":
            instruction = f"Translate the text to {tgt_name}:  {text}"
        else:
            instruction = f"Translate the text to English:  {text}"

        prompt = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        enc = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            gen = self.model.generate(
                enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            )

        raw_out = self.tokenizer.decode(gen[0, enc.shape[1]:], skip_special_tokens=True).strip()

        # Remove leading language tag if present
        if raw_out.startswith(tgt_tag):
            raw_out = raw_out[len(tgt_tag):].strip()
        elif raw_out.startswith("<eng>"):
            raw_out = raw_out[5:].strip()

        # 2. Extract <noun> brackets and resolve substitutions
        def _resolve_bracket(term: str) -> str:
            # Check user-supplied custom glossary first
            if custom_glossary and term.lower() in custom_glossary:
                return custom_glossary[term.lower()]

            # Query multi-candidate lexicon
            cands = self.lookup.get_candidates(term, tgt_code)
            if not cands:
                return term
            if len(cands) == 1 or self.reranker is None:
                return cands[0]

            # Contextual reranking across candidates
            return self.reranker.rerank(raw_out, term, cands)

        final_translated = BracketPostProcessor.substitute(raw_out, _resolve_bracket)
        return final_translated
