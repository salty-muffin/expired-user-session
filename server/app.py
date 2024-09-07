import os
import io
from flask import Flask, Response, abort, request, send_file
import json
from dotenv import load_dotenv

from stt import load_whisper, transcribe_audio
from tts import load_hubert, load_bark, clone_voice, speak, convert_audio_to_mp3

load_dotenv()

app = Flask(__name__)

voice = ""


@app.route("/")
def hello_world():
    return "Expired User Session"


@app.route("/contact", methods=["GET", "POST"])
def contact():
    global voice

    # check for auth
    if not (
        request.authorization
        and request.authorization.parameters["username"] == os.getenv("USERNM")
        and request.authorization.parameters["password"] == os.getenv("PASSWD")
    ):
        abort(401)
    # post request first for sending the question and voice
    if request.method == "POST":
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
        speech = convert_audio_to_mp3(speak(voice, message))

        return send_file(
            speech, mimetype="audio/mpeg", download_name="speech.mp3", max_age=30
        )

    # get requests after for getting continous answers
    if request.method == "GET":
        # check if a voice has been cloned
        if not voice:
            return Response(json.dumps({"error": "No voice found"}), 500)

        message = "This is the next message."
        speech = convert_audio_to_mp3(speak(voice, message))

        return send_file(
            speech, mimetype="audio/mpeg", download_name="speech.mp3", max_age=30
        )


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
