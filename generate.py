# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

from bark.api import generate_audio
from bark.generation import (
    SAMPLE_RATE,
    preload_models,
    # codec_decode,
    # generate_coarse,
    # generate_fine,
    # generate_text_semantic,
)

from scipy.io.wavfile import write as write_wav


semantic_path = "models/semantic_output/pytorch_model.bin"  # set to None if you don't want to use finetuned semantic
coarse_path = "models/coarse_output/pytorch_model.bin"  # set to None if you don't want to use finetuned coarse
fine_path = "models/fine_output/pytorch_model.bin"  # set to None if you don't want to use finetuned fine
use_gpu = False

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


# simple generation
text_prompt = "Hello, my name is Serpy. And, uh — and I like pizza. [laughs]"
voice_name = "speaker_embeddings/en_speaker_0.npz"  # use your custom voice name here if you have on

filepath = "output/generated.wav"

print(f"generating '{text_prompt}' with speaker '{voice_name}'...")
audio_array = generate_audio(
    text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7
)
write_wav(filepath, SAMPLE_RATE, audio_array)
