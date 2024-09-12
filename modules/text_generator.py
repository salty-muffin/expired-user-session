# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import pipeline, set_seed, Pipeline

generator: Pipeline | None = None


def load_generator(model="gpt2-medium", device: str | None = None) -> None:
    global generator

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    print(f"Using {f'cuda:{device}' if device > -1 else 'cpu'} for text generation.")

    generator = pipeline("text-generation", model=model, device=device)


def set_generator_seed(seed: int) -> None:
    set_seed(seed)


def generate(prompt: str, **kwargs) -> str:
    return generator(prompt, pad_token_id=generator.tokenizer.eos_token_id, **kwargs)[
        0
    ]["generated_text"].strip()
