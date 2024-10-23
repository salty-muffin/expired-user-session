import os
from server.stt import Whisper

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")


def test():
    stt = Whisper("openai/whisper-small", use_float16=True)

    message, language = stt.transcribe_audio("server/temp/message.wav")

    print(f"language: '{language}'")
    print(f"transcription: '{message}'")


if __name__ == "__main__":
    test()
