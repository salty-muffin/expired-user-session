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

from prompts import question_prompt, continuation_prompt

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
    gpt_model: str,
    whisper_model: str,
    bark_model: str,
    bark_text_temp: float,
    bark_wave_temp: float,
    use_float16: bool,
    cpu_offload: bool,
    gpt_temp: float | None,
    gpt_top_k: int | None,
    gpt_top_p: float | None,
    gpt_device_map: str | None,
    gpt_bfloat16: bool,
    gpt_do_sample: bool | None,
    wtpsplit_model: str,
) -> None:
    """Generates the responses to be sent to the user. To be called in a seperate process."""

    from huggingface_hub import login

    from stt import Whisper
    from tts import VoiceCloner, Bark
    from text_generator import TextGenerator
    from sentence_splitter import SentenceSplitter

    def next_response(
        gpt_temp: float | None,
        gpt_top_k: int | None,
        gpt_top_p: float | None,
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
                do_sample=gpt_do_sample,
            )
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

    try:
        if huggingface_token := os.environ.get("HF_TOKEN"):
            login(huggingface_token)

        stt = Whisper(whisper_model)
        cloner = VoiceCloner()
        tts = Bark(bark_model, use_float16=use_float16, cpu_offload=cpu_offload)
        text_generator = TextGenerator(
            gpt_model, bfloat16=gpt_bfloat16, device_map=gpt_device_map
        )
        sentence_splitter = SentenceSplitter(wtpsplit_model, "cpu")

        models_ready.set()

        seed = 0

        while not exiting.is_set():
            # wait until a user connects
            users_connected.wait()

            # get the path to the message audio
            message_path = receive_message.recv()

            # transcribe message
            message = stt.transcribe_audio(message_path)
            print(f"Received message: '{message}'.")

            # clone voice
            voice = cloner.clone(message_path)

            # get seed
            if receive_seed.poll():
                seed = receive_seed.recv()
                text_generator.set_seed(seed)

            text, responses = next_response(gpt_temp, gpt_top_k, gpt_top_p, message)
            # generate responses while no new message has been received and users are connected
            while not receive_message.poll():
                print(f"Voicing response: '{text}' (seed: {seed})...")
                # print(f"Voicing response: '{text}' (seed: {seed})...", end=" ")

                speech_data = tts.generate(
                    voice,
                    text,
                    text_temp=bark_text_temp,
                    waveform_temp=bark_wave_temp,
                )

                # (
                #     semantic_generation_config,
                #     coarse_generation_config,
                #     fine_generation_config,
                #     inputs,
                # ) = tts.preprocess(voice, text)
                # print(f"Preprocessed.", end=" ")

                # # exit early if new message has been received
                # if receive_message.poll():
                #     break

                # semantic_output = tts.generate_semantic(
                #     inputs, semantic_generation_config, temperature=bark_text_temp
                # )
                # print(f"Semantic.", end=" ")

                # # exit early if new message has been received
                # if receive_message.poll():
                #     break

                # coarse_output = tts.generate_course(
                #     inputs,
                #     semantic_output,
                #     semantic_generation_config,
                #     coarse_generation_config,
                #     temperature=bark_wave_temp,
                # )
                # print(f"Course.", end=" ")

                # # exit early if new message has been received
                # if receive_message.poll():
                #     break

                # fine_output = tts.generate_fine(
                #     inputs,
                #     coarse_output,
                #     semantic_generation_config,
                #     coarse_generation_config,
                #     fine_generation_config,
                # )
                # print(f"Fine.", end=" ")

                # # exit early if new message has been received
                # if receive_message.poll():
                #     break

                # speech_data = tts.decode(fine_output)
                # print(f"Decoded.")

                # exit early if new message has been received
                if receive_message.poll():
                    break

                send_response.send(speech_data)

                text, responses = next_response(
                    gpt_temp, gpt_top_k, gpt_top_p, message, responses
                )
    except KeyboardInterrupt:
        pass


# fmt: off
@click.command()
@click.option("--gpt_model", type=str, required=True,                      help="The transformer model for speech generation.")
@click.option("--whisper_model", type=str, required=True,                  help="The whisper model for speech transcription.")
@click.option("--bark_model", type=str, required=True,                     help="The bark model for text to speech.")
@click.option("--bark_text_temp", type=click.FloatRange(0.0), default=0.7, help="Temperature for the bark generation (text).")
@click.option("--bark_wave_temp", type=click.FloatRange(0.0), default=0.7, help="Temperature for the bark generation (waveform).")
@click.option("--use_float16", is_flag=True,                               help="Whether to use float16 instead of float32 for bark text to speech (lower vram usage, shorter inference time, quality degradation).")
@click.option("--cpu_offload", is_flag=True,                               help="Whether to offload unused models to the cpu for bark text to speech (lower vram usage, longer inference time).")
@click.option("--gpt_temp", type=click.FloatRange(0.0),                    help="The value used to modulate the next token probabilities.")
@click.option("--gpt_top_k", type=click.IntRange(0),                       help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
@click.option("--gpt_top_p", type=click.FloatRange(0.0),                   help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
@click.option("--gpt_device_map", type=str,                                help="Device map for text generation.")
@click.option("--gpt_bfloat16", is_flag=True, default=False,               help="Use bfloat16 for text generation.")
@click.option("--gpt_do_sample", is_flag=True, default=False,              help="Enable decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling.")
@click.option("--wtpsplit_model", type=str, required=True,                 help="The wtpsplit model for sentence splitting.")
# fmt: on
def respond(**kwarks) -> None:
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
