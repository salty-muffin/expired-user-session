import os
import click
import eventlet
import socketio
from threading import Thread, Lock
from dotenv import load_dotenv
import nltk

from stt import load_whisper, transcribe_audio
from tts import load_hubert, load_bark, clone_voice, speak, convert_audio_to_mp3
from text_generator import load_generator, set_generator_seed, generate

from prompts import question_prompt, continuation_prompt

load_dotenv()

sio = socketio.Server(ping_timeout=60)
app = socketio.WSGIApp(
    sio,
    static_files={
        "/": "/public",
    },
)

voice = ""

responses = []

speech_thread: Thread | None = None

cuda_lock = Lock()

streaming = False

users = set()

click_kwargs = {}


@sio.event
def connect(sid: str, _: dict[str, any], auth: dict[str, str]) -> None:
    if not auth["password"] == os.getenv("PASSWD") or len(users):
        return False

    users.add(sid)
    print(f"Contact established with '{sid}'.")


@sio.event
def disconnect(sid: str) -> None:
    users.remove(sid)
    print(f"Contact lost with '{sid}'.")


@sio.event
def contact(_: str, data: bytes) -> None:
    global streaming, speech_thread

    # stop text generation
    streaming = False

    # write the mp3 data to disk as file
    os.makedirs("temp", exist_ok=True)
    sound_path = os.path.join("temp", "message.mp3")
    with open(sound_path, "wb") as f:
        f.write(data)

    # transcribe the audio
    with cuda_lock:
        message = transcribe_audio(sound_path)
    print(f"Received message: '{message}'.")

    # clone voice
    with cuda_lock:
        voice = clone_voice(sound_path)

    # wait for previous generation to finish
    if speech_thread:
        speech_thread.join()

    # start generating responses
    streaming = True
    speech_thread = sio.start_background_task(
        target=stream_responses, voice=voice, message=message
    )


def stream_responses(voice: str, message: str) -> None:
    text_queue = []
    first_response = True
    # if this is first generation
    if message:
        text_queue = generate_next_response(message)
    while streaming and len(users):
        if not len(text_queue):
            # if nothing is in queue, regenerate
            text_queue = generate_next_response()
            # use this opportunity to quit, if not required to continue
            if not streaming:
                break

        # get next item for tts
        text = text_queue.pop(0)
        print(f"Voicing response: '{text}'.")
        # generate speech
        with cuda_lock:
            try:
                speech_data = speak(voice, text, silent=click_kwargs["silent"])
            except Exception:
                speech_data = []
                sio.send("Please try again.")
                sio.sleep(1)
        # if successful, send to client
        if speech_data is not None:
            mp3 = convert_audio_to_mp3(speech_data)

            sio.emit("first_response" if first_response else "response", mp3.read())
            sio.sleep(click_kwargs["wait"])
            first_response = False


def generate_next_response(message: str | None = None) -> str:
    global responses

    if message:
        responses = []

        prompt = question_prompt.format(message)
    else:
        prompt = continuation_prompt.format(" ".join(responses))

    response_lines = (
        generate(
            prompt,
            max_new_tokens=128,
            do_sample=True,
        )
        .replace(prompt, "")
        .split("\n")
    )
    response_lines = [line.strip() for line in response_lines if line]
    response = response_lines[0]
    responses.append(response)

    return nltk.sent_tokenize(response)


@sio.event
def seed(_: str, data: dict[str, int]) -> None:
    print(f"Received seed: {data['seed']}.")
    set_generator_seed(data["seed"])


def run_socketio() -> None:
    """Function to handle the SocketIO server"""

    eventlet.wsgi.server(eventlet.listen(("", 5000)), app)


# fmt: off
@click.command()
@click.option("--model",  type=str, required=True,                 help="The transformer model for speech generation.")
@click.option("--silent", is_flag=True,                            help="Don't output voice generation progress bars.")
@click.option("--wait",   type=click.FloatRange(1.0), default=1.0, help="Waittime after each socketio emit.")
# fmt: on
def respond(**kwargs) -> None:
    global streaming, speech_thread, click_kwargs

    click_kwargs = kwargs

    load_whisper()
    load_hubert()
    load_bark()
    load_generator(kwargs["model"])

    nltk.download("punkt_tab")

    # start socket connection
    socketio_thread = Thread(target=run_socketio)
    socketio_thread.start()
    try:
        socketio_thread.join()
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        streaming = False

        if speech_thread:
            speech_thread.join()


if __name__ == "__main__":
    respond()