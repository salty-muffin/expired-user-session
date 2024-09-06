# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import os
import io
import numpy as np
import torch
import torchaudio

from bark.api import generate_audio
from bark.generation import (
    SAMPLE_RATE,
    preload_models,
    # codec_decode,
    # generate_coarse,
    # generate_fine,
    # generate_text_semantic,
)

from encodec import EncodecModel
from encodec.utils import convert_audio
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer

from pydub import AudioSegment
import wave

hubert_device: str | None = None
bark_device: str | None = None

hubert_model: CustomHubert | None = None
tokenizer: CustomTokenizer | None = None
encodec_model: EncodecModel | None = None


def load_hubert(large_quant_model=False, device: str | None = None) -> None:
    global hubert_model, tokenizer, encodec_model, hubert_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert_device = device

    print(f"Using {device} for hubert voice cloning.")

    model = (
        ("quantifier_V1_hubert_base_ls960_23.pth", "tokenizer_large.pth")
        if large_quant_model
        else ("quantifier_hubert_base_ls960_14.pth", "tokenizer.pth")
    )

    hubert_model = CustomHubert(
        HuBERTManager.make_sure_hubert_installed(), device=device
    )

    tokenizer = CustomTokenizer.load_from_checkpoint(
        HuBERTManager.make_sure_tokenizer_installed(
            model=model[0], local_file=model[1]
        ),
        device,
    )

    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model.to(device)


def load_bark(model_path="models", device: str | None = None) -> None:
    global bark_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bark_device = device

    print(f"Using {device} for bark text to speech.")

    semantic_path = os.path.join(
        model_path, "semantic_output/pytorch_model.bin"
    )  # set to None if you don't want to use finetuned semantic
    coarse_path = os.path.join(
        model_path, "coarse_output/pytorch_model.bin"
    )  # set to None if you don't want to use finetuned coarse
    fine_path = os.path.join(
        model_path, "fine_output/pytorch_model.bin"
    )  # set to None if you don't want to use finetuned fine

    # download and load all models
    use_gpu = "cuda" in device
    preload_models(
        text_use_gpu=use_gpu,
        text_use_small=False,
        text_model_path=semantic_path,
        coarse_use_gpu=use_gpu,
        coarse_use_small=False,
        coarse_model_path=coarse_path,
        fine_use_gpu=use_gpu,
        fine_use_small=False,
        fine_model_path=fine_path,
        codec_use_gpu=use_gpu,
        force_reload=False,
        path=model_path,
    )


def clone_voice(
    audio_file: str | io.BytesIO,
    voice_outpath=os.path.join("temp", "echo.npz"),
) -> str:
    """Clone the audio from a sample. The sample should be max. 13 seconds long."""

    wav, sr = torchaudio.load(audio_file)

    wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
    wav = wav.to(hubert_device)

    semantic_vectors = hubert_model.forward(
        wav, input_sample_hz=encodec_model.sample_rate
    )
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    # extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav.unsqueeze(0))
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


def convert_audio_to_mp3(audio_data: np.ndarray) -> io.BytesIO:
    """Convert the recorded audio data to a MP3 file and return a file object."""

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes per sample (16-bit PCM)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    audio = AudioSegment.from_wav(wav_io)

    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)

    return mp3_io


def speak(voice_path: str, text: str, text_temp=0.7, waveform_temp=0.7) -> np.ndarray:
    return generate_audio(
        text,
        history_prompt=voice_path,
        text_temp=text_temp,
        waveform_temp=waveform_temp,
    )
