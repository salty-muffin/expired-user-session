from typing import Optional
from threading import Thread

import os
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO
import multiprocessing as mp


class SeanceServer:
    """Main server class handling WebSocket connections and audio streaming."""

    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)

        # Initialize SocketIO with Flask
        self.sio = SocketIO(self.app, ping_timeout=60)

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

        self.profile_pipe = mp.Pipe()

        # Set up the two pages (client interface & dead profiles page)
        self.setup_client_page()
        self.setup_dead_profiles_page()
        self.setup_profile_control()

    def setup_client_page(self):
        """Setup routes for the client page and configure its SocketIO event handlers."""

        # Configure static file routes
        @self.app.route("/")
        def index():
            return send_from_directory("../client/dist", "index.html")

        @self.app.route("/favicon.png")
        def favicon():
            return send_from_directory("../client/dist", "favicon.png")

        @self.app.route("/assets/<path:path>")
        def assets(path):
            return send_from_directory("../client/dist/assets", path)

        # Configure SocketIO event handlers
        @self.sio.on("connect")
        def connect(auth):
            """Handle new client connections with authentication."""
            sid = request.sid

            if not auth.get("password") == os.getenv("PASSWORD"):
                raise ConnectionRefusedError("Authentication failed.")
            if len(self.users):
                raise ConnectionRefusedError("Only one user at a time.")

            self.users.add(sid)
            self.users_connected.set()
            print(f"Contact established with '{sid}'.")

        @self.sio.on("disconnect")
        def disconnect():
            """Handle client disconnections."""
            sid = request.sid

            if sid in self.users:
                self.users.remove(sid)
            if not len(self.users):
                self.users_connected.clear()
            print(f"Contact lost with '{sid}'.")

        @self.sio.on("contact")
        def contact(data):
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

        @self.sio.on("seed")
        def seed(data):
            """Handle random seed updates for response generation."""
            print(f"Received seed: {data['seed']}.")
            self.seed_pipe[1].send(data["seed"])

    def setup_dead_profiles_page(self) -> None:
        """Setup routes for the dead profiles page."""

        @self.app.route("/profiles")
        def profiles():
            return send_from_directory("../dead-profiles/dist", "index.html")

        @self.app.route("/profiles/<path:path>")
        def profiles_assets(path):
            return send_from_directory("../dead-profiles/dist", path)

    def setup_profile_control(self) -> None:
        """Setup endpoints for going back and forwards through the profiles on the dead-profiles page and on the server."""

        @self.app.post("/control")
        def control():
            data = request.json
            print(f"Switched to profile No. {data['index']}: {data['path']}.")

            self.profile_pipe[1].send(data)

            return {"message": "OK"}, 200

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
