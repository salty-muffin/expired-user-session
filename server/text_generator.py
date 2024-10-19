# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import os
import torch
from transformers import pipeline, set_seed


class TextGenerator:
    def __init__(self, model_name, device: str | None = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using {device} for text generation.")

        self._generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
        )

    def set_seed(self, seed: int) -> None:
        set_seed(seed)

    def generate(self, prompt: str, **kwargs) -> str:
        return self._generator(
            prompt, pad_token_id=self._generator.tokenizer.eos_token_id, **kwargs
        )[0]["generated_text"].strip()
