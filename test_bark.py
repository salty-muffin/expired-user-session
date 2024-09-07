from scipy.io.wavfile import write as write_wav

from server.stt import load_whisper, transcribe_audio
from server.tts import (
    load_hubert,
    load_bark,
    clone_voice,
    speak,
    convert_audio_to_mp3,
    SAMPLE_RATE,
)


if __name__ == "__main__":
    load_whisper()
    load_hubert()
    load_bark()

    # input_file = "temp/rec.wav"
    input_file = "temp/rec.mp3"
    output_file_wav = "temp/out.wav"
    output_file_mp3 = "temp/out.mp3"
    voice_file = "temp/echo.npz"

    print("Transcribing...")
    text = transcribe_audio(input_file)
    # text = "Hello, my name is Serpy. And, uh — and I like pizza. [laughs]"
    text = "Hello, this is a very exhausting test."
    # text = "Hello, I have a question. The question is, how are you? Can you please respond?"
    print(text)

    print("Cloning...")
    voice = clone_voice(input_file, voice_file)

    print("Speaking...")
    echo = speak(voice, text)

    print("Saving wav...")
    write_wav(output_file_wav, SAMPLE_RATE, echo)

    print("Saving mp3...")
    mp3_io = convert_audio_to_mp3(echo)
    with open(output_file_mp3, "wb") as f:
        f.write(mp3_io.getbuffer())
