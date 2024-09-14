import subprocess as sp

from modules.tts import load_hubert, load_bark, clone_voice, speak, convert_audio_to_mp3


def test(text: str, text_temp: float, waveform_temp: float) -> None:
    load_hubert()
    load_bark()

    speech_data = speak(
        "temp/echo.npz",
        text,
        text_temp=text_temp,
        waveform_temp=waveform_temp,
        silent=False,
    )
    mp3 = convert_audio_to_mp3(speech_data)
    with open("temp/test.mp3", "wb") as f:
        f.write(mp3.getbuffer())


if __name__ == "__main__":
    test(
        "It is a place full of people who want to look good for their children — children who want more than their parents ever wanted.",
        1.0,
        0.6,
    )
