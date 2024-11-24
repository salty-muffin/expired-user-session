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

from typing import Optional, List, Dict, Any, Tuple
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
        def connect(sid: str, _: Dict[str, Any], auth: Dict[str, str]) -> None:
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
        def seed(_: str, data: Dict[str, int]) -> None:
            """Handle random seed updates for response generation."""
            print(f"Received seed: {data['seed']}.")
            self.seed_pipe[1].send(data["seed"])

    def stream_responses(self) -> None:
        """Stream AI-generated responses back to the client."""
        from audio import convert_audio_to_mp3

        while not self.exiting.is_set():
            response = self.response_pipe[0].recv()
            mp3_data = convert_audio_to_mp3(response, sample_rate=24000)
            event = "first_response" if self.first_response else "response"
            self.sio.emit(event, mp3_data.read())
            self.first_response = False
            self.sio.sleep(1)


class AIProcessor:
    """Handles AI model loading and inference in a separate process."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AI processor with configuration.

        Args:
            config: Dictionary containing model configurations and parameters
        """
        self.config = config
        self.languages = config["languages"]
        self.default_lang = config["default_language"]
        self.translate = config["translate"]

        # Load prompts
        with open(config["prompts"]) as f:
            self.prompts = yaml.safe_load(f)

        # Load AI models
        self.load_models()

    def load_models(self):
        """Load all required AI models."""
        from stt import Whisper
        from tts import VoiceCloner, Bark
        from text_generator import TextGenerator
        from sentence_splitter import SentenceSplitter

        self.stt = Whisper(
            self.config["whisper_model"], multilang=len(self.languages) > 1
        )
        self.voice_cloner = VoiceCloner()
        self.tts = Bark(self.config["bark_model"])
        self.text_generator = TextGenerator(self.config["gpt_model"])
        self.sentence_splitter = SentenceSplitter(self.config["wtpsplit_model"])

    def generate_next_response(
        self,
        language: str,
        message: Optional[str] = None,
        previous_responses: List[str] = [],
    ) -> Tuple[str, List[str]]:
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
            prompt, max_new_tokens=128, **self.config.get("gpt_kwargs", {})
        )[len(prompt) :].strip()

        # Split into sentences and select one
        sentences = self.sentence_splitter.split(full_response)
        selected_sentence = sentences[random.randint(0, len(sentences) - 1)]

        # Update response history
        previous_responses.append(full_response)

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
        models_ready.set()
        current_seed = 0

        while not exiting.is_set():
            # Wait for user connection
            users_connected.wait()

            # Get new message
            message_path = receive_message.recv()

            # Transcribe audio
            message, detected_langs = self.stt.transcribe_audio(message_path)
            current_lang = detected_langs[0] if detected_langs else self.default_lang
            print(f"Received message: '{message}' in language: {current_lang}")

            # Clone voice
            voice = self.voice_cloner.clone(message_path)

            # Update seed if new one received
            if receive_seed.poll():
                current_seed = receive_seed.recv()
                self.text_generator.set_seed(current_seed)

            # Generate responses until new message arrives
            responses: List[str] = []
            while not receive_message.poll():
                # Generate next response
                text, responses = self.generate_next_response(
                    current_lang, message if not responses else None, responses
                )

                print(f"Voicing response: '{text}' (seed: {current_seed})")

                # Generate speech
                speech_data = self.tts.generate(
                    voice, text, **self.config.get("bark_kwargs", {})
                )

                # Send response if no new message
                if not receive_message.poll():
                    send_response.send(speech_data)


@click.command()
@click.option("--whisper_model", required=True, help="Whisper model name")
@click.option("--gpt_model", required=True, help="LLM model name")
@click.option("--bark_model", required=True, help="Bark model name")
@click.option("--wtpsplit_model", required=True, help="Sentence splitting model")
@click.option("--languages", default=["english"], help="Supported languages")
@click.option("--default_language", default="english", help="Default language")
@click.option("--translate", is_flag=True, help="Enable translation")
@click.argument("prompts", type=click.Path(exists=True))
def main(**config):
    """
    Start the Seance application.

    This creates both the WebSocket server and the AI processing components,
    connecting them through pipes for message passing. The AI processor runs
    continuously, generating responses until interrupted by new messages.
    """
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
