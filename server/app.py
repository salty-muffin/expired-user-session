import os
import io
from flask import Flask, Response, abort, request, send_file
import json
from threading import Thread, Lock
from queue import Queue
from dotenv import load_dotenv

from stt import load_whisper, transcribe_audio
from tts import load_hubert, load_bark, clone_voice, speak, convert_audio_to_mp3

load_dotenv()

app = Flask(__name__)

voice = ""

speech_thread: Thread | None = None
speech_queue: Queue | None = None

lock = Lock()


@app.route("/")
def hello_world():
    return Response(json.dumps({"message": "Expired User Session"}), 200)


@app.route("/contact", methods=["GET", "POST"])
def contact():
    global voice, speech_thread, speech_queue

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

        # transcribe the audio
        with lock:
            message = transcribe_audio(file_path)
        print(
            f"received message from {request.authorization.parameters['username']}: {message}"
        )
        with lock:
            voice = clone_voice(file_path)

        # wait for previous generation to finish
        if speech_thread and speech_thread.is_alive():
            speech_thread.join()
        # reset the speech queue
        speech_queue = Queue()

        # generate the response
        response = message

        speak_response(response, voice)

        # start generation on a new response
        speech_thread = Thread(
            target=speak_response, args=(generate_next_response(), voice)
        )
        speech_thread.start()

        return send_file(
            speech_queue.get(),
            mimetype="audio/mpeg",
            download_name="speech.mp3",
            max_age=30,
        )

    # get requests after for getting continous answers
    if request.method == "GET":
        # check if a voice has been cloned
        if not voice:
            return Response(json.dumps({"error": "No voice found"}), 500)

        # wait for previous generation to finish
        if speech_thread and speech_thread.is_alive():
            speech_thread.join()

        # if no response is avaliable generate a new one while blocking
        # otherwise just send an avaliable one and start generating the next one in the background
        if speech_queue.empty():
            # generate a new response while blocking
            speak_response(generate_next_response(), voice)
        else:
            # start generation on a new response in the background
            speech_thread = Thread(
                target=speak_response, args=(generate_next_response(), voice)
            )
            speech_thread.start()

        return send_file(
            speech_queue.get(),
            mimetype="audio/mpeg",
            download_name="speech.mp3",
            max_age=30,
        )


index = 0
words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def generate_next_response() -> str:
    global index
    index += 1
    if index > 5:
        index = 1
    return f"This is message number {words[index]}."


def speak_response(text: str, voice: str) -> io.BytesIO:
    print(f"voicing response: {text}")
    with lock:
        speech_data = speak(voice, text)
    if speech_data is not None:
        speech_queue.put(convert_audio_to_mp3(speech_data))


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
