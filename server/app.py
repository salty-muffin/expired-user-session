# types
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

import os
import click
import eventlet
import socketio
from dotenv import load_dotenv
import random
import multiprocessing as mp
from threading import Thread

from audio import convert_audio_to_mp3

# load and set environment variables
load_dotenv()
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), "models")

# socketio
sio = socketio.Server(ping_timeout=60)
app = socketio.WSGIApp(
    sio,
    static_files={
        "/": "../client/dist/index.html",
        "/favicon.png": "../client/dist/favicon.png",
        "/assets": "../client/dist/assets",
    },
)

# socketio variables
users = set()

# multiprocessing communication objects
receive_message, send_message = mp.Pipe()
receive_response, send_response = mp.Pipe()
receive_seed, send_seed = mp.Pipe()
users_connected = mp.Event()
exiting = mp.Event()
models_ready = mp.Event()

# constants
SAMPLE_RATE = 24_000

# variables
background_thread: Thread | None = None
first_response = True


@sio.event
def connect(sid: str, _: dict[str, any], auth: dict[str, str]) -> None:
    """Gets called when a client connects."""

    if not auth["password"] == os.getenv("PASSWORD"):
        raise ConnectionRefusedError("Authentication failed.")
    if len(users):
        raise ConnectionRefusedError("Only one user at a time.")

    users.add(sid)
    users_connected.set()

    print(f"Contact established with '{sid}'.")


@sio.event
def disconnect(sid: str) -> None:
    """Gets called when a client disconnect."""

    if sid in users:
        users.remove(sid)
    if not len(users):
        users_connected.clear()

    print(f"Contact lost with '{sid}'.")


@sio.event
def contact(_: str, data: bytes) -> None:
    """Gets called when a client sends a voice message."""

    global first_response, background_thread

    # write the sound data to disk as file
    os.makedirs("temp", exist_ok=True)
    message_path = os.path.join("temp", "message.wav")
    with open(message_path, "wb") as f:
        f.write(data)

    # send over the file path
    first_response = True
    send_message.send(message_path)

    # start thread to stream responses
    if not background_thread:
        background_thread = sio.start_background_task(target=send_responses)


def send_responses() -> None:
    """Sends the generated responses to the user. To be called in a seperate thread in the main process."""

    global first_response

    while not exiting.is_set():
        response = receive_response.recv()
        mp3 = convert_audio_to_mp3(response, SAMPLE_RATE)
        sio.emit("first_response" if first_response else "response", mp3.read())
        first_response = False
        sio.sleep(1)


@sio.event
def seed(_: str, data: dict[str, int]) -> None:
    """Gets called when a client sends seed."""

    print(f"Received seed: {data['seed']}.")
    send_seed.send(data["seed"])


def generate_responses(
    receive_message: Connection,
    send_response: Connection,
    receive_seed: Connection,
    users_connected: Event,
    models_ready: Event,
    exiting: Event,
    **kwargs,
) -> None:
    """Generates the responses to be sent to the user. To be called in a seperate process."""

    import yaml
    from collections import Counter
    from huggingface_hub import login

    from stt import Whisper
    from tts import VoiceCloner, Bark
    from text_generator import TextGenerator
    from sentence_splitter import SentenceSplitter

    # get languages & prompts
    languages = kwargs.pop("languages")
    default_lang = kwargs.pop("default_language")
    prompts = {}
    prompts_dir = kwargs.pop("prompts_dir")
    for lang in languages:
        with open(os.path.join(prompts_dir, f"{lang}.yml")) as file:
            prompts[lang] = yaml.safe_load(file)

    def filter_kwargs_by_prefix(prefix, kwargs, remove_none=False):
        """
        Filters keys from the kwargs that match the given prefix and removes the prefix from the keys.

        :param prefix: The prefix to match and remove.
        :param kwargs: The dictionary to filter.
        :param remove_none: If True, filter out keys with None values.
        :return: A new dictionary with filtered and renamed keys.
        """
        filtered_kwargs = {
            key[len(prefix) :]: value
            for key, value in kwargs.items()
            if key.startswith(prefix)
        }

        if remove_none:
            filtered_kwargs = {
                key: value
                for key, value in filtered_kwargs.items()
                if value is not None
            }

        return filtered_kwargs

    # get kwargs for each model
    whisper_kwargs = filter_kwargs_by_prefix("whisper_", kwargs, remove_none=True)
    gpt_kwargs = filter_kwargs_by_prefix("gpt_", kwargs, remove_none=True)
    bark_kwargs = filter_kwargs_by_prefix("bark_", kwargs, remove_none=True)
    wtpsplit_kwargs = filter_kwargs_by_prefix("wtpsplit_", kwargs, remove_none=True)

    def next_response(
        language: str, message: str | None = None, responses=[], **kwargs
    ) -> str:
        # use fallback, if language is not provided
        if not language in languages:
            language = default_lang

        if message:
            responses = []

            prompt = prompts[language]["question_prompt"].format(message)
        else:
            prompt = prompts[language]["continuation_prompt"].format(
                " ".join(responses)
            )

        response_lines = (
            text_generator.generate(prompt, max_new_tokens=128, **kwargs)
            .replace(prompt, "")
            .split("\n")
        )
        response_lines = [line.strip() for line in response_lines if line]
        if not len(response_lines):
            return "..."
        response = response_lines[0]
        responses.append(response)

        sentences = sentence_splitter.split(response)
        sentence = sentences[random.randint(0, len(sentences) - 1)]
        return sentence, responses

    def find_language(languages: list[str]) -> str:
        return Counter(languages).most_common(1)[0][0]

    try:
        if huggingface_token := os.environ.get("HUGGINGFACE_TOKEN"):
            login(huggingface_token)

        stt = Whisper(
            whisper_kwargs.pop("model"),
            multilang=len(languages) > 1,
            use_float16=whisper_kwargs.pop("use_float16"),
        )
        cloner = VoiceCloner()
        tts = Bark(
            bark_kwargs.pop("model"),
            use_float16=bark_kwargs.pop("use_float16"),
            cpu_offload=bark_kwargs.pop("cpu_offload"),
        )
        text_generator = TextGenerator(
            gpt_kwargs.pop("model"),
            device_map=gpt_kwargs.pop("device_map", None),
            use_bfloat16=gpt_kwargs.pop("use_bfloat16"),
        )
        sentence_splitter = SentenceSplitter(wtpsplit_kwargs.pop("model"), "cpu")

        models_ready.set()

        seed = 0

        while not exiting.is_set():
            # wait until a user connects
            users_connected.wait()

            # get the path to the message audio
            message_path = receive_message.recv()

            # transcribe message
            message, langs = stt.transcribe_audio(message_path)
            current_lang = find_language(langs) if langs else default_lang
            print(f"Received message: '{message}' in language: {current_lang}.")

            # clone voice
            voice = cloner.clone(message_path)

            # get seed
            if receive_seed.poll():
                seed = receive_seed.recv()
                text_generator.set_seed(seed)

            text, responses = next_response(current_lang, message, **gpt_kwargs)
            # generate responses while no new message has been received and users are connected
            while not receive_message.poll():
                print(f"Voicing response: '{text}' (seed: {seed})...")

                speech_data = tts.generate(voice, text, **bark_kwargs)

                # exit early if new message has been received
                if receive_message.poll():
                    break

                send_response.send(speech_data)

                text, responses = next_response(
                    current_lang, message, responses, **gpt_kwargs
                )
    except KeyboardInterrupt:
        pass


def parse_comma_list(s: list | str) -> list[str]:
    if isinstance(s, list):
        return s
    return [e.strip() for e in s.split(",")]


# fmt: off
@click.command()
# whisper options
@click.option("--whisper_model", type=str, required=True,                         help="The whisper model for speech transcription")
@click.option("--whisper_use_float16", is_flag=True, default=False,               help="Whether to use float16 instead of float32 for whisper speech to text (lower vram usage, shorter inference time, possible quality degradation)")
# text generation options
@click.option("--gpt_model", type=str, required=True,                             help="The transformer model for speech generation")
@click.option("--gpt_temperature", type=click.FloatRange(0.0),                    help="The value used to modulate the next token probabilities")
@click.option("--gpt_top_k", type=click.IntRange(0),                              help="The number of highest probability vocabulary tokens to keep for top-k-filtering")
@click.option("--gpt_top_p", type=click.FloatRange(0.0),                          help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation")
@click.option("--gpt_do_sample", is_flag=True, default=None,                      help="Enable decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling")
@click.option("--gpt_use_bfloat16", is_flag=True, default=False,                  help="Load the model as bfloat16 instead of float32")
@click.option("--gpt_device_map", type=click.Choice(["auto"]),                    help="When set to 'auto', automatically fills all available space on the GPU(s) first, then the CPU, and finally, the hard drive")
# text to speech options
@click.option("--bark_model", type=str, required=True,                            help="The bark model for text to speech")
@click.option("--bark_semantic_temperature", type=click.FloatRange(0.0),          help="Temperature for the bark generation (semantic/text)")
@click.option("--bark_coarse_temperature", type=click.FloatRange(0.0),            help="Temperature for the bark generation (course waveform)")
@click.option("--bark_fine_temperature", type=click.FloatRange(0.0), default=0.5, help="Temperature for the bark generation (fine waveform)")
@click.option("--bark_use_float16", is_flag=True, default=False,                  help="Whether to use float16 instead of float32 for bark text to speech (lower vram usage, shorter inference time, quality degradation)")
@click.option("--bark_cpu_offload", is_flag=True, default=False,                  help="Whether to offload unused models to the cpu for bark text to speech (lower vram usage, longer inference time)")
# sentence splitting options
@click.option("--wtpsplit_model", type=str, required=True,                        help="The wtpsplit model for sentence splitting")
# language options
@click.option("--languages", type=parse_comma_list, default=["english"],          help="The languages to accept as inputs (stt, tts, text generation & sentence splitting models need to be able to work with the languages provided)")
@click.option("--default_language", type=str, default="english",                  help="The fallback language in case the detected language is not provided")
# prompts
@click.argument("prompts_dir", type=click.Path(exists=True, file_okay=False))
# fmt: on
def respond(**kwarks) -> None:
    # check if prompts for all languages exist
    for lang in kwarks["languages"]:
        fn = os.path.join(kwarks["prompts_dir"], f"{lang}.yml")
        if not os.path.isfile(fn):
            raise click.FileError(
                fn, f"No prompts file for language '{lang}' was provided"
            )

    # start response process
    response_process = mp.Process(
        target=generate_responses,
        args=(
            receive_message,
            send_response,
            receive_seed,
            users_connected,
            models_ready,
            exiting,
        ),
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

        if response_process:
            response_process.join()
        if background_thread:
            background_thread.join()


if __name__ == "__main__":
    respond()
