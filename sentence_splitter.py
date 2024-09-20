import torch
from wtpsplit import SaT


class SentenceSplitter:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using {device} for sentence splitting.")

        self._model = SaT(model_name)

        if "cuda" in device:
            self._model.half().to(device)

    def split(self, text: str) -> list[str]:
        return self._model.split(text)
