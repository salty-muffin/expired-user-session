import os
import click
from pynput import keyboard
import sounddevice as sd
import numpy as np
import wave
from threading import Thread
import io
from pydub import AudioSegment
import requests
import subprocess
import time
from dotenv import load_dotenv

# settings for audio recording
dtype = np.int16  # data type (16-bit PCM)

recording = False
audio_data = []

streaming = False

stream_thread: Thread | None = None
record_thread: Thread | None = None

click_kwargs = {}

load_dotenv()


def record_audio() -> None:
    """Record audio while recording flag is True."""

    global audio_data

    samplerate = click_kwargs["samplerate"]

    audio_data = []

    print("Recording started...")

    # callback function to append recorded data
    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        audio_data.append(indata.copy())

    # start recording stream
    with sd.InputStream(
        samplerate=samplerate, channels=1, dtype=dtype, callback=callback
    ) as _:
        while recording:
            sd.sleep(100)  # wait a little while recording

    print("Recording stopped.")


def on_press(key: keyboard.Key) -> None:
    """Start recording when the 'space' key is pressed."""
    global recording, streaming, record_thread

    if key == keyboard.Key.space and not recording:
        # start recording but stop the playbackstream if it's running
        recording = True
        streaming = False

        record_thread = Thread(target=record_audio)
        record_thread.start()


def on_release(key: keyboard.Key) -> bool | None:
    """Stop recording when the 'space' key is released."""

    global recording, index

    endpoint = click_kwargs["endpoint"]

    if key == keyboard.Key.space and recording:
        recording = False

        if audio_data:
            mp3 = convert_audio_to_mp3(np.concatenate(audio_data, axis=0))
            send_message_to_url(mp3, endpoint)
        else:
            print("Something went wrong with the recording.")


def convert_audio_to_mp3(audio_data: np.ndarray) -> io.BytesIO:
    """Convert the recorded audio data to a MP3 file and return a file object."""

    samplerate = click_kwargs["samplerate"]

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes per sample (16-bit PCM)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    audio = AudioSegment.from_wav(wav_io)

    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)

    return mp3_io


def stream_responses(url: str) -> None:
    """Get the current response and play it back."""

    global streaming

    breaktime = click_kwargs["breaktime"]

    while streaming:
        # break out of the stream loop if streaming stops
        print(f"Waiting for {breaktime} seconds...")
        start_time = time.time()
        time.sleep(0.01)
        while time.time() - start_time <= breaktime:
            if not streaming:
                break
            time.sleep(0.01)

        # send the GET request for voices
        response = requests.get(url, auth=(os.getenv("USERNM"), os.getenv("PASSWD")))
        print(f"Asked for response from '{url}'.")
        if response.status_code == 200:
            # write the mp3 data to disk as file
            os.makedirs("temp", exist_ok=True)
            sound_path = os.path.join("temp", "response.mp3")
            with open(sound_path, "wb") as f:
                f.write(response.content)

            # play back the file
            playback = subprocess.Popen(
                ["ffplay", "-v", "0", "-nodisp", "-autoexit", sound_path]
            )

            # wait until playback is finished or streaming stops
            while playback.poll() is None:
                if not streaming:
                    playback.terminate()
                time.sleep(0.1)
        else:
            print(f"Error: {response.status_code} - {response.text}")


def send_message_to_url(audio: io.BytesIO, url: str) -> None:
    """Send the audio to an URL as a a file object."""

    global streaming, stream_thread

    # create a dictionary for the file to be sent in the POST request
    files = {"file": ("message.mp3", audio, "audio/mpeg")}

    # send the POST request
    response = requests.post(
        url, files=files, auth=(os.getenv("USERNM"), os.getenv("PASSWD"))
    )
    print(f"Message sent to '{url}'.")

    # check if the request was successful
    if response.status_code == 200:
        streaming = True

        # start thread for continously streaming answers
        stream_thread = Thread(target=stream_responses, args=[url])
        stream_thread.start()
    else:
        print(f"Error: {response.status_code} - {response.text}")


# fmt: off
@click.command()
@click.option("--samplerate", type=int,   default=44100, help="The recording sample rate.")
@click.option("--endpoint",   type=str,   required=True, help="The endpoint to send the recordings to.")
@click.option("--breaktime",  type=float, default=0.0,   help="Time between requesting new voices in seconds.")
# fmt: on
def contact(**kwargs) -> None:
    global streaming, recording, click_kwargs

    click_kwargs = kwargs

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
    finally:
        recording = False
        streaming = False

        if stream_thread and stream_thread.is_alive():
            stream_thread.join()
        if record_thread and record_thread.is_alive():
            record_thread.join()


if __name__ == "__main__":
    contact()
