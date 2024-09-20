import subprocess as sp

from tts import Bark
from audio import convert_audio_to_mp3


def test(
    bark_model: str,
    use_float16: bool,
    cpu_offload: bool,
    text: str,
    text_temp: float,
    waveform_temp: float,
) -> None:
    tts = Bark(bark_model, use_float16=use_float16, cpu_offload=cpu_offload)

    speech_data = tts.generate(
        "temp/echo.npz",
        text,
        text_temp=text_temp,
        waveform_temp=waveform_temp,
    )
    mp3 = convert_audio_to_mp3(speech_data)
    with open("temp/test.mp3", "wb") as f:
        f.write(mp3.getbuffer())


if __name__ == "__main__":
    test(
        "suno/bark",
        False,
        False,
        "It is a place full of people who want to look good for their children — children who want more than their parents ever wanted.",
        1.0,
        0.6,
    )
