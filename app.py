# types
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

import os
import click
import eventlet
import socketio
from dotenv import load_dotenv
import nltk
import random
import multiprocessing as mp
from threading import Thread

import stt
import tts
from tts import convert_audio_to_mp3
import text_generator

from prompts import question_prompt, continuation_prompt

load_dotenv()

sio = socketio.Server(ping_timeout=60)
app = socketio.WSGIApp(
    sio,
    static_files={
        "/": "./client/dist/index.html",
        "/favicon.png": "./client/dist/favicon.png",
        "/favicon.svg": "./client/dist/favicon.svg",
        "/assets": "./client/dist/assets",
    },
)

# multiprocessing communication objects
receive_message, send_message = mp.Pipe()
receive_response, send_response = mp.Pipe()
users_connected = mp.Event()
exiting = mp.Event()
models_ready = mp.Event()

# socketio variables
users = set()

# variables
background_thread: Thread | None = None
first_response = True


@sio.event
def connect(sid: str, _: dict[str, any], auth: dict[str, str]) -> None:
    """Gets called when a client connects."""

    global background_thread

    if not auth["password"] == os.getenv("PASSWORD"):
        raise ConnectionRefusedError("Authentication failed.")
    if len(users):
        raise ConnectionRefusedError("Only one user at a time.")

    users.add(sid)
    users_connected.set()

    print(f"Contact established with '{sid}'.")

    # start thread to stream responses
    background_thread = sio.start_background_task(target=send_responses)


def send_responses() -> None:
    """Sends the generated responses to the user. To be called in a seperate thread in the main process."""

    global first_response

    while users_connected.is_set() and not exiting.is_set():
        response = receive_response.recv()
        mp3 = convert_audio_to_mp3(response)
        sio.emit("first_response" if first_response else "response", mp3.read())
        first_response = False
        sio.sleep(1)


@sio.event
def disconnect(sid: str) -> None:
    """Gets called when a client disconnect."""

    users.remove(sid)
    if not len(users):
        users_connected.clear()

        if background_thread:
            background_thread.join()

    print(f"Contact lost with '{sid}'.")


@sio.event
def contact(_: str, data: bytes) -> None:
    """Gets called when a client sends a voice message."""

    global first_response

    # write the sound data to disk as file
    os.makedirs("temp", exist_ok=True)
    message_path = os.path.join("temp", "message.wav")
    with open(message_path, "wb") as f:
        f.write(data)

    # send over the file path
    first_response = True
    send_message.send(message_path)


@sio.event
def seed(_: str, data: dict[str, int]) -> None:
    """Gets called when a client sends seed."""

    print(f"Received seed: {data['seed']}.")
    text_generator.set_generator_seed(data["seed"])


def generate_responses(
    receive: Connection,
    send: Connection,
    users_connected: Event,
    models_ready: Event,
    gpt_model: str,
    whisper_model: str,
    silent: bool,
    bark_text_temp: float,
    bark_wave_temp: float,
    gpt_temp: float,
    gpt_top_k: int,
    gpt_top_p: float,
) -> None:
    """Generates the responses to be sent to the user. To be called in a seperate process."""

    stt.load_whisper(whisper_model)
    tts.load_hubert()
    tts.load_bark()
    text_generator.load_generator(gpt_model)

    models_ready.set()

    while not exiting.is_set():
        # wait until a user connects
        users_connected.wait()

        # get the path to the message audio
        message_path = receive.recv()

        # transcribe message
        message = stt.transcribe_audio(message_path)
        print(f"Received message: '{message}'.")

        # clone voice
        voice = tts.clone_voice(message_path)

        text, responses = next_response(gpt_temp, gpt_top_k, gpt_top_p, message)
        # generate responses while no new message has been received and users are connected
        while not receive.poll():
            print(f"Voicing response: '{text}'.")

            speech_data = tts.generate(
                voice,
                text,
                text_temp=bark_text_temp,
                waveform_temp=bark_wave_temp,
                silent=silent,
            )

            send.send(speech_data)


def next_response(
    gpt_temp: float,
    gpt_top_k: int,
    gpt_top_p: float,
    message: str | None = None,
    responses=[],
) -> str:
    if message:
        responses = []

        prompt = question_prompt.format(message)
    else:
        prompt = continuation_prompt.format(" ".join(responses))

        response_lines = (
            text_generator.generate(
                prompt,
                temperature=gpt_temp,
                top_k=gpt_top_k,
                top_p=gpt_top_p,
                max_new_tokens=128,
                do_sample=True,
            )
            .replace(prompt, "")
            .split("\n")
        )
    response_lines = [line.strip() for line in response_lines if line]
    if not len(response_lines):
        return "..."
    response = response_lines[0]
    responses.append(response)

    sentences = nltk.sent_tokenize(response)
    sentence = sentences[random.randint(0, len(sentences) - 1)]
    return sentence, responses


# fmt: off
@click.command()
@click.option("--gpt_model", type=str, required=True,                      help="The transformer model for speech generation.")
@click.option("--whisper_model", type=str, default="base",                 help="The whisper model for speech transcription.")
@click.option("--silent", is_flag=True,                                    help="Don't output voice generation progress bars.")
@click.option("--bark_text_temp", type=click.FloatRange(0.0), default=0.7, help="Temperature for the bark generation (text).")
@click.option("--bark_wave_temp", type=click.FloatRange(0.0), default=0.7, help="Temperature for the bark generation (waveform).")
@click.option("--gpt_temp", type=click.FloatRange(0.0), default=1.0,       help="The value used to modulate the next token probabilities.")
@click.option("--gpt_top_k", type=click.IntRange(0), default=50,           help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
@click.option("--gpt_top_p", type=click.FloatRange(0.0), default=1.0,      help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
# fmt: on
def respond(**kwarks) -> None:
    # download tokenization data for sentance splitting
    nltk_path = os.path.join(os.getcwd(), "nltk_data")
    nltk.download("punkt_tab", download_dir=nltk_path)
    os.environ["NLTK_DATA"] = nltk_path

    # start response process
    response_process = mp.Process(
        target=generate_responses,
        args=(receive_message, send_response, users_connected, models_ready),
        kwargs=kwarks,
    )
    response_process.start()

    # wait until the models are loaded
    models_ready.wait()

    # start socket connection
    try:
        eventlet.wsgi.server(eventlet.listen(("", 5000)), app)
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
    finally:
        exiting.set()

        response_process.join()
        background_thread.join()


if __name__ == "__main__":
    respond()
