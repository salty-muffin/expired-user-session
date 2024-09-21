# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import os
import torch
from transformers import pipeline


class Whisper:
    def __init__(self, model_name, device: str | None = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using {device} for whisper speech to text.")

        self._model = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )

    def transcribe_audio(self, path: str) -> str:
        """Transcribes the audio with whisper."""

        return self._model(path, batch_size=8)["text"].strip()
