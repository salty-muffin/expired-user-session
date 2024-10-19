# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import torch
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Whisper:
    def __init__(
        self, model_name, use_float16=False, device: str | None = None
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        torch_dtype = (
            torch.float16 if "cuda" in device and use_float16 else torch.float32
        )

        print(
            f"Using {device} with {'float16' if torch_dtype == torch.float16 else 'float32'} for whisper speech to text."
        )

        self._transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            generate_kwargs={"task": "transcribe"},
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            device=device,
        )

    def transcribe_audio(self, path: str) -> tuple[str, list[str]]:
        """
        Transcribes the audio with whisper.

        Returns a tuple of (transcription, [languages])
        """

        output = self._transcriber(path, batch_size=8, return_language=True)

        return output["text"].strip(), [chunk["language"] for chunk in output["chunks"]]
