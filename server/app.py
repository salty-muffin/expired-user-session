import os
import io
from flask import Flask, Response, abort, request, send_file
import json
from dotenv import load_dotenv

from stt import load_whisper, transcribe_audio
from tts import load_hubert, load_bark, clone_voice, speak, convert_audio_to_mp3

load_dotenv()

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Expired User Session"


@app.post("/contact")
def contact():
    # check for auth
    if not (
        request.authorization
        and request.authorization.parameters["username"] == os.getenv("USERNM")
        and request.authorization.parameters["password"] == os.getenv("PASSWD")
    ):
        abort(401)
    # check if file is included
    if not "file" in request.files:
        return Response(json.dumps({"error": "No file part"}), 400)

    # save file temporarily
    file = request.files["file"]
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    message = transcribe_audio(file_path)
    print(
        f"received message from {request.authorization.parameters['username']}: {message}"
    )
    voice = clone_voice(file_path)
    echo = convert_audio_to_mp3(speak(voice, message))

    return send_file(echo, mimetype="audio/mpeg", download_name="echo.mp3", max_age=30)


@app.errorhandler(401)
def unauthorized(error):
    return Response(
        "Wrong user and/or passcode",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'},
    )


if __name__ == "__main__":
    load_whisper()
    load_hubert()
    load_bark()

    app.run()
