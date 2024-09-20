import torch
from wtpsplit import SaT


class SentenceSplitter:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using {device} for sentence splitting.")
        split_model_name = model_name.split("/")
        self._model = SaT(split_model_name[1], hub_prefix=split_model_name[0])

        if "cuda" in device:
            self._model.half().to(device)

    def split(self, text: str) -> list[str]:
        return self._model.split(text)
