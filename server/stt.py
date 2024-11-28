# Filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import pipeline


class Whisper:
    def __init__(
        self,
        model_name: str,
        multilang=False,
        dtype="default",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "default": None,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        if dtype not in dtype_map.keys():
            raise ValueError(
                f"dtype for {type(self).__name__} (transformers) only accepts {dtype_map.keys()}"
            )
        torch_dtype = dtype_map[dtype] if "cuda" in device else dtype_map["default"]

        print(
            f"Using {device} with {dtype} for whisper speech to text with '{model_name}'."
        )

        generate_kwargs = {"task": "transcribe"} if multilang else None

        self._transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            generate_kwargs=generate_kwargs,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            device=device,
        )

    def transcribe_audio(self, path: str) -> tuple[str, list[str], None]:
        """
        Transcribes the audio with whisper.

        Returns a tuple of (transcription, [languages])
        """

        output = self._transcriber(path, batch_size=8, return_language=True)

        return output["text"].strip(), [
            chunk["language"] for chunk in output["chunks"] if chunk["language"]
        ]
