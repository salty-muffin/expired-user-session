# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import os
import io
import numpy as np
import torch
import torchaudio

from transformers import BarkProcessor, BarkModel

from encodec import EncodecModel
from encodec.utils import convert_audio
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer


class VoiceCloner:
    def __init__(self, large_quant_model=False, device: str | None = None) -> None:
        model = (
            ("quantifier_V1_hubert_base_ls960_23.pth", "tokenizer_large.pth")
            if large_quant_model
            else ("quantifier_hubert_base_ls960_14.pth", "tokenizer.pth")
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        print(f"Using {device} for hubert voice cloning.")

        self._hubert_model = CustomHubert(
            HuBERTManager.make_sure_hubert_installed(), device=self._device
        )
        self._tokenizer = CustomTokenizer.load_from_checkpoint(
            HuBERTManager.make_sure_tokenizer_installed(
                model=model[0], local_file=model[1]
            ),
            self._device,
        )
        self._encodec_model = EncodecModel.encodec_model_24khz()
        self._encodec_model.set_target_bandwidth(6.0)
        self._encodec_model.to(self._device)

    def clone(
        self,
        audio_file: str | io.BytesIO,
        voice_outpath=os.path.join("temp", "echo.npz"),
    ) -> str:
        # load and pre-process the audio waveform
        wav, sr = torchaudio.load(audio_file)

        wav = convert_audio(
            wav, sr, self._encodec_model.sample_rate, self._encodec_model.channels
        )
        wav = wav.to(self._device)

        semantic_vectors = self._hubert_model.forward(
            wav, input_sample_hz=self._encodec_model.sample_rate
        )
        semantic_tokens = self._tokenizer.get_token(semantic_vectors)

        # extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self._encodec_model.encode(wav.unsqueeze(0))
        codes = torch.cat(
            [encoded[0] for encoded in encoded_frames], dim=-1
        ).squeeze()  # [n_q, T]

        # move codes to cpu
        codes = codes.cpu().numpy()
        # move semantic tokens to cpu
        semantic_tokens = semantic_tokens.cpu().numpy()

        np.savez(
            voice_outpath,
            fine_prompt=codes,
            coarse_prompt=codes[:2, :],
            semantic_prompt=semantic_tokens,
        )

        return voice_outpath


class Bark:
    def __init__(self, device: str | None = None, use_float16=False) -> None:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self._device = device

        print(
            f"Using {f'cuda:{device}' if device > -1 else 'cpu'} for bark text to speech."
        )

        self._processor = BarkProcessor.from_pretrained("suno/bark")
        self._model = (
            BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(
                self._device
            )
            if use_float16 and self._device >= 0
            else BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(
                self._device
            )
        )

        if self._device >= 0:
            self._model = self._model.to_bettertransformer()

    @property
    def sample_rate(self) -> int:
        return self._model.generation_config.sample_rate

    def generate(
        self, voice_path: str, text: str, text_temp=0.7, waveform_temp=0.7
    ) -> np.ndarray:
        inputs = self._processor(text, voice_preset=voice_path).to(self._device)

        audio_array = self._model.generate(**inputs)
        return audio_array.cpu().numpy().squeeze()
