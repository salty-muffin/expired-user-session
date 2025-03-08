import os
import socketio
import multiprocessing as mp
from threading import Thread

from typing import Optional, Any


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
