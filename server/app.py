"""
Seance: An AI-powered voice communication system

This application creates an interactive "seance" experience where users can communicate
with an AI that responds with synthesized speech. It uses several AI models to:
1. Convert user speech to text
2. Generate contextual responses and follow-up messages
3. Convert responses back to speech using the user's voice characteristics

The system runs a WebSocket server to handle real-time communication and runs the AI
models in a separate process to maintain responsiveness. When a user asks a question,
the AI will continue generating related responses until a new question is asked.
"""

import os
import click
import eventlet
import socketio
import random
from dotenv import load_dotenv
import multiprocessing as mp
from threading import Thread
import yaml
from collections import Counter

from huggingface_hub import login

from typing import Optional, Any
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

# Initialize environment
load_dotenv()
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), "models")


class SeanceServer:
    """Main server class handling WebSocket connections and audio streaming."""

    def __init__(self):
        # Initialize SocketIO server
        self.sio = socketio.Server(ping_timeout=60)
        self.app = socketio.WSGIApp(
            self.sio,
            static_files={
                "/": "../client/dist/index.html",
                "/favicon.png": "../client/dist/favicon.png",
                "/assets": "../client/dist/assets",
            },
        )

        # Connection management
        self.users = set()
        self.background_thread: Optional[Thread] = None
        self.first_response = True

        # Multiprocessing communication
        self.message_pipe = mp.Pipe()
        self.response_pipe = mp.Pipe()
        self.seed_pipe = mp.Pipe()
        self.users_connected = mp.Event()
        self.exiting = mp.Event()
        self.models_ready = mp.Event()

        # Set up event handlers
        self.setup_handlers()

    def setup_handlers(self):
        """Configure SocketIO event handlers."""

        @self.sio.event
        def connect(sid: str, _: dict[str, Any], auth: dict[str, str]) -> None:
            """Handle new client connections with authentication."""

            if not auth["password"] == os.getenv("PASSWORD"):
                raise ConnectionRefusedError("Authentication failed.")
            if len(self.users):
                raise ConnectionRefusedError("Only one user at a time.")

            self.users.add(sid)
            self.users_connected.set()
            print(f"Contact established with '{sid}'.")

        @self.sio.event
        def disconnect(sid: str) -> None:
            """Handle client disconnections."""

            if sid in self.users:
                self.users.remove(sid)
            if not len(self.users):
                self.users_connected.clear()
            print(f"Contact lost with '{sid}'.")

        @self.sio.event
        def contact(_: str, data: bytes) -> None:
            """Handle incoming voice messages."""

            # Save received audio
            os.makedirs("temp", exist_ok=True)
            message_path = os.path.join("temp", "message.wav")
            with open(message_path, "wb") as f:
                f.write(data)

            # Process message
            self.first_response = True
            self.message_pipe[1].send(message_path)

            # Start response streaming if needed
            if not self.background_thread:
                self.background_thread = self.sio.start_background_task(
                    target=self.stream_responses
                )

        @self.sio.event
        def seed(_: str, data: dict[str, int]) -> None:
            """Handle random seed updates for response generation."""
            print(f"Received seed: {data['seed']}.")
            self.seed_pipe[1].send(data["seed"])

    def stream_responses(self) -> None:
        """Stream AI-generated responses back to the client."""
        from audio import convert_audio_to_mp3

        while not self.exiting.is_set():
            if self.response_pipe[0].poll():
                response = self.response_pipe[0].recv()
                mp3_data = convert_audio_to_mp3(response, sample_rate=24_000)
                event = "first_response" if self.first_response else "response"
                self.sio.emit(event, mp3_data.read())
                self.first_response = False
            self.sio.sleep(1)


def filter_config_by_prefix(prefix, config, remove_none=False):
    """
    Filters keys from the config that match the given prefix and removes the prefix from the keys.

    :param prefix: The prefix to match and remove.
    :param config: The dictionary to filter.
    :param remove_none: If True, filter out keys with None values.
    :return: A new dictionary with filtered and renamed keys.
    """
    filtered_config = {
        key[len(prefix) :]: value
        for key, value in config.items()
        if key.startswith(prefix)
    }

    if remove_none:
        filtered_config = {
            key: value for key, value in filtered_config.items() if value is not None
        }

    return filtered_config


def find_language(languages: list[str]) -> str:
    """
    Finds the most common occurance if a language in a list of language name strings.
    """

    return Counter(languages).most_common(1)[0][0]


def remove_config_items(config: dict[str, Any], keys: list[str]) -> None:
    """
    Removes specified keys from the given configuration dictionary in-place.

    Args:
        config: Dictionary to modify. Will be mutated by this function.
        keys: List of keys to remove from the config dictionary. Keys that don't
            exist in the config will be silently ignored.
    """

    for key in keys:
        config.pop(key, None)


class AIProcessor:
    """Handles AI model loading and inference in a separate process."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize AI processor with configuration.

        Args:
            config: Dictionary containing model configurations and parameters
        """

        self.config = config
        self.languages: list[str] = self.config.pop("languages")
        self.default_lang: str = self.config.pop("default_language")
        self.translate: bool = self.config.pop("translate")

        # Load prompts
        with open(config.pop("prompts")) as f:
            self.prompts = yaml.safe_load(f)

        # Validate translation config parameters
        if self.default_lang not in self.languages:
            raise RuntimeError(
                f"Default language not present in languages. Please change the languages."
            )
        if self.default_lang not in self.prompts.keys():
            raise RuntimeError(
                f"No prompts for default language '{self.default_lang}' were provided"
            )
        if not self.translate:
            for lang in self.languages:
                if lang not in self.prompts.keys():
                    raise RuntimeError(
                        f"No prompts for language '{lang}' were provided"
                    )

    def load_models(self):
        """Load all required AI models."""

        from stt import Whisper
        from tts import VoiceCloner, Bark
        from text_generator import (
            TextGenerator,
            TextGeneratorCTranslate,
        )
        from sentence_splitter import SentenceSplitter
        from translator import Opus, OpusCTranslate2

        # Get config for each model
        self.whisper_config = filter_config_by_prefix(
            "whisper_", self.config, remove_none=True
        )
        self.bark_config = filter_config_by_prefix(
            "bark_", self.config, remove_none=True
        )
        self.gpt_config = filter_config_by_prefix("gpt_", self.config, remove_none=True)
        self.wtpsplit_config = filter_config_by_prefix(
            "wtpsplit_", self.config, remove_none=True
        )
        self.opus_config = filter_config_by_prefix(
            "opus_", self.config, remove_none=True
        )

        # Log into huggingface in case access limited models need to be fetched
        if huggingface_token := os.environ.get("HF_TOKEN"):
            login(huggingface_token)

        # Setup whisper text to speech
        self.stt = Whisper(
            self.whisper_config.pop("model"),
            multilang=len(self.languages) > 1,
            dtype=self.whisper_config.pop("dtype"),
        )

        # Setup voice cloning for bark
        self.voice_cloner = VoiceCloner()
        self.tts = Bark(
            self.bark_config.pop("model"),
            dtype=self.bark_config.pop("dtype"),
            use_better_transformer=self.bark_config.pop("use_better_transformer"),
            cpu_offload=self.bark_config.pop("cpu_offload"),
        )

        # Setup text generation
        if "ctranslate_dir" in self.gpt_config.keys():
            self.text_generator = TextGeneratorCTranslate(
                self.gpt_config.pop("model"),
                ctranslate_dir=self.gpt_config.pop("ctranslate_dir"),
                dtype=self.gpt_config.pop("dtype"),
                activation_scales=self.gpt_config.pop("activation_scales", None),
            )
            remove_config_items(self.gpt_config, ["device_map", "compression"])
        else:
            self.text_generator = TextGenerator(
                self.gpt_config.pop("model"),
                dtype=self.gpt_config.pop("dtype"),
                device_map=self.gpt_config.pop("device_map", None),
            )
            remove_config_items(self.gpt_config, ["activation_scales", "compression"])

        # Setup sentence splitting
        self.sentence_splitter = SentenceSplitter(
            self.wtpsplit_config.pop("model"), "cpu"
        )

        # Only load opus translation models if translation is enabled
        if self.translate:
            if "ctranslate_dir" in self.opus_config.keys():
                self.translator = OpusCTranslate2(
                    self.opus_config.pop("model_names_base"),
                    self.languages,
                    ctranslate_dir=self.opus_config.pop("ctranslate_dir"),
                    dtype=self.opus_config.pop("dtype"),
                    device="cpu",
                )
            else:
                self.translator = Opus(
                    self.opus_config.pop("model_names_base"),
                    self.languages,
                    dtype=self.opus_config.pop("dtype"),
                    device="cpu",
                )

    def generate_next_response(
        self,
        language: str,
        message: Optional[str] = None,
        previous_responses: list[str] = [],
    ) -> tuple[str, list[str]]:
        """
        Generate the next response in the conversation.

        Args:
            language: Current conversation language
            message: New user message (if any)
            previous_responses: List of previous AI responses

        Returns:
            Tuple of (selected sentence, updated responses list)
        """
        # Use fallback language if needed
        if language not in self.languages:
            language = self.default_lang

        # Generate prompt based on context
        if message:
            previous_responses = []
            prompt = self.prompts[language]["question_prompt"].format(message)
        else:
            prompt = self.prompts[language]["continuation_prompt"].format(
                " ".join(previous_responses)
            )

        # Generate response
        full_response = self.text_generator.generate(
            prompt, max_new_tokens=128, **self.gpt_config
        )[len(prompt) :].strip()
        if not len(full_response):
            return "..."

        response = full_response.splitlines()[0]

        # Update response history
        previous_responses.append(response)

        # Split into sentences and select one
        sentences = self.sentence_splitter.split(response)
        selected_sentence = sentences[random.randint(0, len(sentences) - 1)]

        return selected_sentence, previous_responses

    def run(
        self,
        receive_message: Connection,
        send_response: Connection,
        receive_seed: Connection,
        users_connected: Event,
        models_ready: Event,
        exiting: Event,
    ):
        """
        Main processing loop that handles messages and generates responses.

        Runs continuously in a separate process, generating initial responses
        to user messages and follow-up responses until interrupted by a new
        message.
        """

        # Load AI models
        self.load_models()

        models_ready.set()
        current_seed = 0

        while not exiting.is_set():
            # Wait for user connection
            users_connected.wait()

            # Get new message
            message_path = receive_message.recv()

            # Transcribe audio
            message, detected_langs = self.stt.transcribe_audio(message_path)
            current_lang = (
                find_language(detected_langs) if detected_langs else self.default_lang
            )
            print(f"Received message: '{message}' in language: {current_lang}")

            # Translate if necessary
            translated_lang = self.default_lang
            if self.translate and current_lang != self.default_lang:
                translated_lang = current_lang
                current_lang = self.default_lang
                message = self.translator.translate(
                    message, translated_lang, self.default_lang
                )
                print(f"Translated message to: '{message}'.")

            # Clone voice
            voice = self.voice_cloner.clone(message_path)

            # Update seed if new one received
            if receive_seed.poll():
                current_seed = receive_seed.recv()
                self.text_generator.set_seed(current_seed)

            # Generate responses until new message arrives
            responses: list[str] = []
            while not receive_message.poll():
                # Generate next response
                text, responses = self.generate_next_response(
                    current_lang, message if not responses else None, responses
                )

                # Translate if necessary
                if self.translate and translated_lang != self.default_lang:
                    print(
                        f"Translating response: '{text}' (target: {translated_lang})..."
                    )
                    text = self.translator.translate(
                        text, self.default_lang, translated_lang
                    )

                # Exit early if new message has been received
                if receive_message.poll():
                    break

                print(f"Voicing response: '{text}' (seed: {current_seed})")

                # Generate speech
                speech_data = self.tts.generate(voice, text, **self.bark_config)

                # Send response if no new message
                if not receive_message.poll():
                    send_response.send(speech_data)


def validate_command_line_arguments(config) -> None:
    """Validate all command line arguments coming through click."""

    if config["gpt_activation_scales"] and not config["gpt_ctranslate_dir"]:
        print(
            "warning: Setting '--gpt_activation_scales' without setting '--gpt_ctranslate_dir' has no effect."
        )


def parse_comma_list(s: list | str) -> list[str]:
    if isinstance(s, list):
        return s
    return [e.strip() for e in s.split(",")]


# fmt: off
@click.command()
# Whisper options
@click.option("--whisper_model", type=str, required=True,                              help="Whisper model for speech transcription")
# @click.option("--whisper_ctranslate_dir", type=click.Path(file_okay=False),            help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--whisper_dtype", type=str, default="default",                          help="Torch dtype to use for the model (transformers: default, float32, float16; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# Text generation options
@click.option("--gpt_model", type=str, required=True,                                  help="Transformer model for speech generation")
@click.option("--gpt_device_map", type=str, default=None,                              help="How to distribute the model across GRPU, CPU & memory (possible options: 'auto')")
@click.option("--gpt_ctranslate_dir", type=click.Path(file_okay=False),                help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--gpt_activation_scales", type=click.Path(exists=True, dir_okay=False), help="Path to the activation scales for converting the model to CTranslate2")
@click.option("--gpt_temperature", type=click.FloatRange(0.0),                         help="Value used to modulate the next token probabilities")
@click.option("--gpt_top_k", type=click.IntRange(0),                                   help="Nmber of highest probability vocabulary tokens to keep for top-k-filtering")
@click.option("--gpt_top_p", type=click.FloatRange(0.0),                               help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation")
@click.option("--gpt_do_sample", is_flag=True, default=None,                           help="Enable decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling")
@click.option("--gpt_dtype", type=str, default="default",                              help="Torch dtype to use for the model (transformers: default, float32, bfloat16; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# Text to speech options
@click.option("--bark_model", type=str, required=True,                                 help="Bark model for text to speech")
@click.option("--bark_semantic_temperature", type=click.FloatRange(0.0),               help="Temperature for the bark generation (semantic/text)")
@click.option("--bark_coarse_temperature", type=click.FloatRange(0.0),                 help="Temperature for the bark generation (course waveform)")
@click.option("--bark_fine_temperature", type=click.FloatRange(0.0), default=0.5,      help="Temperature for the bark generation (fine waveform)")
@click.option("--bark_use_better_transformer", is_flag=True, default=False,            help="Optimize bark with BetterTransformer (shorter inference time)")
@click.option("--bark_dtype", type=str, default="default",                             help="Torch dtype to use for the model (default, float32, float16)")
@click.option("--bark_cpu_offload", is_flag=True, default=False,                       help="Offload unused models to the cpu for bark text to speech (lower vram usage, longer inference time)")
# Sentence splitting options
@click.option("--wtpsplit_model", type=str, required=True,                             help="Wtpsplit model for sentence splitting")
# Language options
@click.option("--languages", type=parse_comma_list, default=["english"],               help="Languages to accept as inputs (stt, tts, text generation & sentence splitting models need to be able to work with the languages provided)")
@click.option("--default_language", type=str, default="english",                       help="Fallback language in case the detected language is not provided")
# Translation
@click.option("--translate", is_flag=True, default=False,                              help="Always translate to and from the default language instead of using a multi language model")
@click.option("--opus_model_names_base", type=str,                                     help="String to be formatted with the corresponding language codes (e.g. 'Helsinki-NLP/opus-mt-{}-{}' -> 'Helsinki-NLP/opus-mt-en-de'), needs trantion to be enabled")
@click.option("--opus_ctranslate_dir", type=click.Path(file_okay=False),               help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--opus_dtype", type=str, default="default",                             help="Torch dtype to use for the model (transformers: default, float32; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# Prompts
@click.argument("prompts", type=click.Path(exists=True, dir_okay=False))
# fmt: on
def main(**config):
    """
    Start the Seance application.

    This creates both the WebSocket server and the AI processing components,
    connecting them through pipes for message passing. The AI processor runs
    continuously, generating responses until interrupted by new messages.
    """

    validate_command_line_arguments(config)

    server = SeanceServer()

    # Start AI processor in separate process
    processor = mp.Process(
        target=AIProcessor(config).run,
        args=(
            server.message_pipe[0],
            server.response_pipe[1],
            server.seed_pipe[0],
            server.users_connected,
            server.models_ready,
            server.exiting,
        ),
    )
    processor.start()

    # Wait for models to load
    server.models_ready.wait()

    # Start server
    try:
        eventlet.wsgi.server(eventlet.listen(("", 5000)), server.app)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.exiting.set()
        if processor.is_alive():
            processor.join()
        if server.background_thread:
            server.background_thread.join()


if __name__ == "__main__":
    main()
