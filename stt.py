# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import torch
import whisper


class Whisper:
    def __init__(self, model_name="base", device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using {device} for whisper speech to text.")

        self._model = whisper.load_model(model_name, device=device)

    def transcribe_audio(self, path: str) -> str:
        """Transcribes the audio with whisper."""

        return self._model.transcribe(path)["text"].strip()
