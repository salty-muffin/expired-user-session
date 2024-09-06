import os
import click
from pynput import keyboard
import sounddevice as sd
import numpy as np
import wave
import threading
import io
from pydub import AudioSegment
import requests
from dotenv import load_dotenv

# settings for audio recording
sr = 0  # sample rate (Hz)
dtype = np.int16  # data type (16-bit PCM)

ep = ""  # endpoint to send the audio to

recording = False
audio_data = []
stream = None

load_dotenv()


def record_audio() -> None:
    """Record audio while recording flag is True."""

    global audio_data, stream
    audio_data = []

    print("Recording started...")

    # callback function to append recorded data
    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        audio_data.append(indata.copy())

    # start recording stream
    with sd.InputStream(
        samplerate=sr, channels=1, dtype=dtype, callback=callback
    ) as stream:
        while recording:
            sd.sleep(100)  # wait a little while recording

    print("Recording stopped.")


def start_recording() -> None:
    """Start a separate thread for recording."""

    global recording
    recording = True

    record_thread = threading.Thread(target=record_audio)
    record_thread.start()


def stop_recording() -> None:
    """Stop the recording and save the audio to a file."""

    global recording, index
    recording = False

    mp3 = convert_audio_to_mp3(np.concatenate(audio_data, axis=0))
    send_audio_to_url(mp3, ep)


def on_press(key: keyboard.Key) -> None:
    """Start recording when the 'space' key is pressed."""

    try:
        if key == keyboard.Key.space:
            if not recording:
                start_recording()
    except AttributeError:
        pass


def on_release(key: keyboard.Key) -> bool | None:
    """Stop recording when the 'space' key is released."""

    if key == keyboard.Key.space and recording:
        stop_recording()
        return False  # stop the listener


def convert_audio_to_mp3(audio_data: np.ndarray) -> io.BytesIO:
    """Convert the recorded audio data to a MP3 file and return a file object."""

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes per sample (16-bit PCM)
        wf.setframerate(sr)
        wf.writeframes(audio_data.tobytes())
    audio = AudioSegment.from_wav(wav_io)

    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)

    return mp3_io


def send_audio_to_url(audio: io.BytesIO, url: str) -> None:
    """Send the audio to an URL as a a file object."""

    # create a dictionary for the file to be sent in the POST request
    files = {"file": ("message.mp3", audio, "audio/mpeg")}

    # send the POST request
    response = requests.post(
        url, files=files, auth=(os.getenv("USERNM"), os.getenv("PASSWD"))
    )
    print(f"Message sent to '{url}'.")

    # check if the request was successful
    if response.status_code == 200:
        # process the response containing the MP3 audio data object
        os.makedirs("temp", exist_ok=True)
        with open(os.path.join("temp", "response.mp3"), "wb") as f:
            f.write(response.content)
    else:
        print(f"Error: {response.status_code} - {response.text}")


# fmt: off
@click.command()
@click.option("--samplerate", type=int, default=44100, help="The recording sample rate.")
@click.option("--endpoint",   type=str, required=True, help="The endpoint to send the recordings to.")
# fmt: on
def contact(samplerate: int, endpoint: str) -> None:
    global sr, ep
    sr = samplerate
    ep = endpoint

    print("Press 'space' to start recording, release to stop.")

    try:
        while True:
            # start key listener in the main loop
            with keyboard.Listener(
                on_press=on_press, on_release=on_release
            ) as listener:
                listener.join()
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")


if __name__ == "__main__":
    contact()
