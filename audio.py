import io
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


def convert_audio_to_mp3(audio_data: np.ndarray, sample_rate: int) -> io.BytesIO:
    """Convert the recorded audio data to a MP3 file and return a file object."""

    wav_io = io.BytesIO()
    mp3_io = io.BytesIO()
    wavfile.write(wav_io, sample_rate, audio_data)
    wav_io.seek(0)
    AudioSegment.from_wav(wav_io).export(mp3_io, format="mp3")

    return mp3_io
