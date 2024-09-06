import os
import io
from flask import Flask, Response, abort, request
import json
from dotenv import load_dotenv
import whisper

load_dotenv()

app = Flask(__name__)

model = whisper.load_model("base")


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


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

    message = process_audio(file_path)

    return Response(json.dumps({"message": message}), 200)


@app.errorhandler(401)
def unauthorized(error):
    return Response(
        "Wrong user and/or passcode",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'},
    )


def process_audio(path: str) -> str:
    """Transcribes the audio and then uses it to generate responses which are returned as speech."""

    result = model.transcribe(path)

    return result["text"]


if __name__ == "__main__":
    app.run()
