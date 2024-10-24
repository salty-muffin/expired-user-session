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
        dtype="default",
        device: str | None = None,
        device_map: Literal["auto"] | None = None,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "default": None,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map.keys():
            raise ValueError(
                f"dtype for {type(self).__name__} (transformers) only accepts {dtype_map.keys()}"
            )
        torch_dtype = dtype_map[dtype] if "cuda" in device else dtype_map["default"]

        print(f"Using {device} with {dtype} for text generation.")

        self._generator = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
            device_map=device_map,
        )

    def set_seed(self, seed: int) -> None:
        set_seed(seed)

    def generate(self, prompt: str, **kwargs) -> str:
        return self._generator(
            prompt, pad_token_id=self._generator.tokenizer.eos_token_id, **kwargs
        )[0]["generated_text"].strip()
