# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import os
import io
import numpy as np
import torch
import torchaudio

from transformers import BarkProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)

from optimum.bettertransformer import BetterTransformer

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
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        print(f"Using {self._device} for hubert voice cloning.")

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
    def __init__(
        self,
        model_name,
        device: str | None = None,
        use_float16=False,
        cpu_offload=False,
    ) -> None:

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._use_float16 = use_float16 and "cuda" in self._device

        print(
            f"Using {self._device} with {'float16' if self._use_float16 else 'float32'} for bark text to speech."
        )

        self._processor: BarkProcessor = BarkProcessor.from_pretrained(model_name)
        self._model: BarkModel = (
            BarkModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(self._device)
            if self._use_float16
            else BarkModel.from_pretrained(model_name).to(self._device)
        )

        if "cuda" in self._device:
            # self._model = BetterTransformer.transform(self._model)
            if cpu_offload:
                self._model.enable_cpu_offload()

    @property
    def sample_rate(self) -> int:
        return self._model.generation_config.sample_rate

    def generate(self, voice_path: str, text: str, **kwargs) -> np.ndarray:
        inputs = self._processor(text, voice_preset=voice_path).to(self._device)

        audio_array = self._model.generate(
            **inputs,
            **kwargs,
        )

        audio_array = audio_array.cpu().numpy().squeeze()

        if self._use_float16:
            audio_array = audio_array.astype(np.float32)

        return audio_array

    def preprocess(self, voice_path: str, text: str):
        """Generate configs."""
        semantic_generation_config = BarkSemanticGenerationConfig(
            **self._model.generation_config.semantic_config
        )
        coarse_generation_config = BarkCoarseGenerationConfig(
            **self._model.generation_config.coarse_acoustics_config
        )
        fine_generation_config = BarkFineGenerationConfig(
            **self._model.generation_config.fine_acoustics_config
        )

        inputs = self._processor(text, voice_preset=voice_path).to(self._device)

        return (
            semantic_generation_config,
            coarse_generation_config,
            fine_generation_config,
            inputs,
        )

    def generate_semantic(self, inputs, semantic_generation_config, temperature=0.7):
        """1. Generate from the semantic model."""

        semantic_output = self._model.semantic.generate(
            inputs["input_ids"],
            history_prompt=inputs["history_prompt"],
            semantic_generation_config=semantic_generation_config,
            temperature=temperature,
        )

        return semantic_output

    def generate_course(
        self,
        inputs,
        semantic_output,
        semantic_generation_config,
        coarse_generation_config,
        temperature=0.7,
    ):
        """2. Generate from the coarse model."""

        coarse_output = self._model.coarse_acoustics.generate(
            semantic_output,
            history_prompt=inputs["history_prompt"],
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            codebook_size=self._model.generation_config.codebook_size,
            temperature=temperature,
        )

        return coarse_output

    def generate_fine(
        self,
        inputs,
        coarse_output,
        semantic_generation_config,
        coarse_generation_config,
        fine_generation_config,
        temperature=0.5,
    ):
        """3. "generate" from the fine model."""

        fine_output = self._model.fine_acoustics.generate(
            coarse_output,
            history_prompt=inputs["history_prompt"],
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            fine_generation_config=fine_generation_config,
            codebook_size=self._model.generation_config.codebook_size,
            temperature=temperature,
        )

        if getattr(self._model, "fine_acoustics_hook", None) is not None:
            # Manually offload fine_acoustics to CPU
            # and load codec_model to GPU
            # since bark doesn't use codec_model forward pass
            self._model.fine_acoustics_hook.offload()
            self._model.codec_model = self.codec_model.to(self.device)

        return fine_output

    def decode(self, fine_output):
        """4. Decode the output and generate audio array."""

        audio_array = self._model.codec_decode(fine_output)

        if getattr(self._model, "codec_model_hook", None) is not None:
            # offload codec_model to CPU
            self._model.codec_model_hook.offload()

        audio_array = audio_array.cpu().numpy().squeeze()

        if self._use_float16:
            audio_array = audio_array.astype(np.float32)

        return audio_array
