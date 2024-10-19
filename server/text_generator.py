from typing import Literal

# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import pipeline, set_seed


class TextGenerator:
    def __init__(
        self,
        model_name,
        device: str | None = None,
        device_map: Literal["auto"] | None = None,
        use_bfloat16: bool = False,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        kwargs = {}
        if use_bfloat16:
            kwargs["torch_dtype"] = torch.bfloat16
        if device_map:
            print(
                f"Using device_map '{device_map}' for text generation{' with bfloat16' if use_bfloat16 else ''}."
            )
            kwargs["device_map"] = device_map
        else:
            print(
                f"Using {device} for text generation{' with bfloat16' if use_bfloat16 else ''}."
            )
            kwargs["device"] = device

        self._generator = pipeline("text-generation", model=model_name, **kwargs)

    def set_seed(self, seed: int) -> None:
        set_seed(seed)

    def generate(self, prompt: str, **kwargs) -> str:
        return self._generator(
            prompt, pad_token_id=self._generator.tokenizer.eos_token_id, **kwargs
        )[0]["generated_text"].strip()
