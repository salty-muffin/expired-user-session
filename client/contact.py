import os
from pynput import keyboard
import sounddevice as sd
import numpy as np
import wave
import threading

# settings for audio recording
samplerate = 44100  # sample rate (Hz)
channels = 1  # number of channels (mono)
dtype = np.int16  # data type (16-bit PCM)

recording = False
audio_data = []
stream = None

index = 0
filename = "output.wav"


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
        samplerate=samplerate, channels=channels, dtype=dtype, callback=callback
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

    _filename = os.path.splitext(filename)
    save_audio_to_wav(
        f"{_filename[0]}_{index}{_filename[1]}", np.concatenate(audio_data, axis=0)
    )
    print(f"Recording saved as '{_filename[0]}_{index}{_filename[1]}'")
    index += 1


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


def save_audio_to_wav(filename: str, audio_data: np.ndarray):
    """Save the recorded audio data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes per sample (16-bit PCM)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())


if __name__ == "__main__":
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
