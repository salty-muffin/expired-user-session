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
    from text_generator import (
        TextGenerator,
        TextGeneratorCTranslate,
        TextGeneratorAirLLM,
    )
    from sentence_splitter import SentenceSplitter
    from translator import Opus, OpusCTranslate2

    # get languages & prompts
    languages = kwargs.pop("languages")
    default_lang = kwargs.pop("default_language")
    with open(kwargs.pop("prompts")) as file:
        prompts: dict = yaml.safe_load(file)

    # check if translation is enabled
    translate = kwargs.pop("translate")
    if translate and len(languages) <= 1:
        translate = False
        print(
            "Warning: 'translate' flag is set but only one language is given. Nothing will be translated."
        )

    # check if prompts for all languages exist
    if default_lang not in languages:
        raise RuntimeError(
            f"Default language not present in languages. Please change the languages."
        )
    if default_lang not in prompts.keys():
        raise RuntimeError(
            f"No prompts for default language '{default_lang}' was provided"
        )
    if not translate:
        for lang in languages:
            if lang not in prompts.keys():
                raise RuntimeError(f"No prompts for language '{lang}' was provided")

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
    opus_kwargs = filter_kwargs_by_prefix("opus_", kwargs, remove_none=True)

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

        response_lines = text_generator.generate(prompt, max_new_tokens=128, **kwargs)[
            len(prompt) : :
        ].split("\n")
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
        if huggingface_token := os.environ.get("HF_TOKEN"):
            login(huggingface_token)

        stt = Whisper(
            whisper_kwargs.pop("model"),
            multilang=len(languages) > 1,
            dtype=whisper_kwargs.pop("dtype"),
        )
        cloner = VoiceCloner()
        tts = Bark(
            bark_kwargs.pop("model"),
            dtype=bark_kwargs.pop("dtype"),
            use_better_transformer=bark_kwargs.pop("use_better_transformer"),
            cpu_offload=bark_kwargs.pop("cpu_offload"),
        )
        if "ctranslate_dir" in gpt_kwargs.keys():
            text_generator = TextGeneratorCTranslate(
                gpt_kwargs.pop("model"),
                ctranslate_dir=gpt_kwargs.pop("ctranslate_dir"),
                dtype=gpt_kwargs.pop("dtype"),
                activation_scales=gpt_kwargs.pop("activation_scales", None),
            )
            gpt_kwargs.pop("device_map", None)
            gpt_kwargs.pop("compression", None)
        elif gpt_kwargs.pop("airllm"):
            text_generator = TextGeneratorAirLLM(
                gpt_kwargs.pop("model"),
                compression=gpt_kwargs.pop("compression"),
            )
            gpt_kwargs.pop("activation_scales", None)
            gpt_kwargs.pop("device_map", None)
            gpt_kwargs.pop("dtype", None)
        else:
            text_generator = TextGenerator(
                gpt_kwargs.pop("model"),
                dtype=gpt_kwargs.pop("dtype"),
                device_map=gpt_kwargs.pop("device_map", None),
            )
            gpt_kwargs.pop("activation_scales", None)
            gpt_kwargs.pop("compression", None)
        sentence_splitter = SentenceSplitter(wtpsplit_kwargs.pop("model"), "cpu")
        # only load opus translation models if translation is enabled
        if translate:
            if "ctranslate_dir" in opus_kwargs.keys():
                translator = OpusCTranslate2(
                    opus_kwargs.pop("model_names_base"),
                    languages,
                    ctranslate_dir=opus_kwargs.pop("ctranslate_dir"),
                    dtype=opus_kwargs.pop("dtype"),
                    device="cpu",
                )
            else:
                translator = Opus(
                    opus_kwargs.pop("model_names_base"),
                    languages,
                    dtype=opus_kwargs.pop("dtype"),
                    device="cpu",
                )

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
            # translate if it should
            translated_lang = default_lang
            if translate and current_lang != default_lang:
                translated_lang = current_lang
                current_lang = default_lang
                message = translator.translate(message, translated_lang, default_lang)
                print(f"Translated message to: '{message}'.")

            # clone voice
            voice = cloner.clone(message_path)

            # get seed
            if receive_seed.poll():
                seed = receive_seed.recv()
                text_generator.set_seed(seed)

            text, responses = next_response(current_lang, message, **gpt_kwargs)
            # generate responses while no new message has been received and users are connected
            while not receive_message.poll():
                if translate and translated_lang != default_lang:
                    print(
                        f"Translating response: '{text}' (target: {translated_lang})..."
                    )
                    text = translator.translate(text, default_lang, translated_lang)

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
@click.option("--whisper_model", type=str, required=True,                              help="Whisper model for speech transcription")
# @click.option("--whisper_ctranslate_dir", type=click.Path(file_okay=False),            help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--whisper_dtype", type=str, default="default",                          help="Torch dtype to use for the model (transformers: default, float32, float16; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# text generation options
@click.option("--gpt_model", type=str, required=True,                                  help="Transformer model for speech generation")
@click.option("--gpt_device_map", type=str, default=None,                              help="How to distribute the model across GRPU, CPU & memory (possible options: 'auto')")
@click.option("--gpt_ctranslate_dir", type=click.Path(file_okay=False),                help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2, mutually exclusive with airLLM)")
@click.option("--gpt_activation_scales", type=click.Path(exists=True, dir_okay=False), help="Path to the activation scales for converting the model to CTranslate2")
@click.option("--gpt_airllm", is_flag=True, default=False,                             help="Use model with airLLM (mutually exclusive with CTranslate2)")
@click.option("--gpt_compression", type=click.Choice(["4bit", "8bit"]), default=None,  help="AirLLM compression")
@click.option("--gpt_temperature", type=click.FloatRange(0.0),                         help="Value used to modulate the next token probabilities")
@click.option("--gpt_top_k", type=click.IntRange(0),                                   help="Nmber of highest probability vocabulary tokens to keep for top-k-filtering")
@click.option("--gpt_top_p", type=click.FloatRange(0.0),                               help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation")
@click.option("--gpt_do_sample", is_flag=True, default=None,                           help="Enable decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling")
@click.option("--gpt_dtype", type=str, default="default",                              help="Torch dtype to use for the model (transformers: default, float32, bfloat16; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# text to speech options
@click.option("--bark_model", type=str, required=True,                                 help="Bark model for text to speech")
@click.option("--bark_semantic_temperature", type=click.FloatRange(0.0),               help="Temperature for the bark generation (semantic/text)")
@click.option("--bark_coarse_temperature", type=click.FloatRange(0.0),                 help="Temperature for the bark generation (course waveform)")
@click.option("--bark_fine_temperature", type=click.FloatRange(0.0), default=0.5,      help="Temperature for the bark generation (fine waveform)")
@click.option("--bark_use_better_transformer", is_flag=True, default=False,            help="Optimize bark with BetterTransformer (shorter inference time)")
@click.option("--bark_dtype", type=str, default="default",                             help="Torch dtype to use for the model (default, float32, float16)")
@click.option("--bark_cpu_offload", is_flag=True, default=False,                       help="Offload unused models to the cpu for bark text to speech (lower vram usage, longer inference time)")
# sentence splitting options
@click.option("--wtpsplit_model", type=str, required=True,                             help="Wtpsplit model for sentence splitting")
# language options
@click.option("--languages", type=parse_comma_list, default=["english"],               help="Languages to accept as inputs (stt, tts, text generation & sentence splitting models need to be able to work with the languages provided)")
@click.option("--default_language", type=str, default="english",                       help="Fallback language in case the detected language is not provided")
# translation
@click.option("--translate", is_flag=True, default=False,                              help="Always translate to and from the default language instead of using a multi language model")
@click.option("--opus_model_names_base", type=str,                                     help="String to be formatted with the corresponding language codes (e.g. 'Helsinki-NLP/opus-mt-{}-{}' -> 'Helsinki-NLP/opus-mt-en-de'), needs trantion to be enabled")
@click.option("--opus_ctranslate_dir", type=click.Path(file_okay=False),               help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--opus_dtype", type=str, default="default",                             help="Torch dtype to use for the model (transformers: default, float32; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# prompts
@click.argument("prompts", type=click.Path(exists=True, dir_okay=False))
# fmt: on
def respond(**kwarks) -> None:
    # check click options
    if kwarks["gpt_ctranslate_dir"] and kwarks["gpt_airllm"]:
        raise click.ClickException(
            "'--gpt_ctranslate_dir' and '--gpt_airllm' are mutually exclusive!"
        )
    if kwarks["gpt_activation_scales"] and not kwarks["gpt_ctranslate_dir"]:
        print(
            "warning: Setting '--gpt_activation_scales' without setting '--gpt_ctranslate_dir' has no effect."
        )
    if kwarks["gpt_compression"] and not kwarks["gpt_airllm"]:
        print(
            "warning: Setting '--gpt_compression' without setting '--gpt_airllm' has no effect."
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
