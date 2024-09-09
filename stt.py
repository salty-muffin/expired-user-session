# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import torch
import whisper

model: whisper.Whisper | None = None


def load_whisper(model_name="base", device: str | None = None) -> None:
    global model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} for whisper speech to text.")

    model = whisper.load_model(model_name, device=device)


def transcribe_audio(path: str) -> str:
    """Transcribes the audio with whisper."""

    if model is None:
        raise RuntimeError("Whisper model must be loaded before transcription")

    result = model.transcribe(path)

    return result["text"].strip()
