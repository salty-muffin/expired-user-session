# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

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

from scipy.io.wavfile import write as write_wav

# --- loading hubert ---

large_quant_model = False  # use the larger pretrained model
device = "cpu"  # 'cuda', 'cpu', 'cuda:0'

model = (
    ("quantifier_V1_hubert_base_ls960_23.pth", "tokenizer_large.pth")
    if large_quant_model
    else ("quantifier_hubert_base_ls960_14.pth", "tokenizer.pth")
)

print("loading HuBERT...")
hubert_model = CustomHubert(HuBERTManager.make_sure_hubert_installed(), device=device)

print("loading quantizer...")
tokenizer = CustomTokenizer.load_from_checkpoint(
    HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]),
    device,
)

print("loading encodec...")
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model.to(device)

# --- loading bark ---

semantic_path = "models/semantic_output/pytorch_model.bin"  # set to None if you don't want to use finetuned semantic
coarse_path = "models/coarse_output/pytorch_model.bin"  # set to None if you don't want to use finetuned coarse
fine_path = "models/fine_output/pytorch_model.bin"  # set to None if you don't want to use finetuned fine
use_gpu = "cuda" in device

# download and load all models
print("loading bark...")
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
    path="models",
)

print("downloaded and loaded models!")

# --- cloning voice ---

# load and pre-process the audio waveform
audio_filepath = "input/audio.wav"  # the audio you want to clone (under 13 seconds)
wav, sr = torchaudio.load(audio_filepath)

print("creating coarse and fine prompts...")
wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
wav = wav.to(device)

print("extracting semantics...")
semantic_vectors = hubert_model.forward(wav, input_sample_hz=encodec_model.sample_rate)
print("tokenizing semantics...")
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

voice_name = "output"  # whatever you want the name of the voice to be
output_path = "speaker_embeddings/custom/" + voice_name + ".npz"
np.savez(
    output_path,
    fine_prompt=codes,
    coarse_prompt=codes[:2, :],
    semantic_prompt=semantic_tokens,
)
print("cloned voice!")


# simple generation
text_prompt = "Hello, my name is Serpy. And, uh — and I like pizza. [laughs]"
voice_name = output_path  # "speaker_embeddings/en_speaker_0.npz"  # use your custom voice name here if you have on

filepath = "output/audio.wav"

print(f"generating '{text_prompt}' with speaker '{voice_name}'...")
audio_array = generate_audio(
    text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7
)
write_wav(filepath, SAMPLE_RATE, audio_array)
