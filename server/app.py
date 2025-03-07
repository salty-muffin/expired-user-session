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

import click
from dotenv import load_dotenv
import multiprocessing as mp

from seance_server import SeanceServer
from ai_processor import AIProcessor

# Initialize environment
load_dotenv()


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
@click.option("--whisper_model", type=str, required=True,                                help="Whisper model for speech transcription")
# @click.option("--whisper_ctranslate_dir", type=click.Path(file_okay=False),            help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--whisper_dtype", type=str, default="default",                            help="Torch dtype to use for the model (transformers: default, float32, float16; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# Text generation options
@click.option("--gpt_model", type=str, required=True,                                    help="Transformer model for speech generation")
@click.option("--gpt_device_map", type=str, default=None,                                help="How to distribute the model across GRPU, CPU & memory (possible options: 'auto')")
@click.option("--gpt_ctranslate_dir", type=click.Path(file_okay=False),                  help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--gpt_activation_scales", type=click.Path(exists=True, dir_okay=False),   help="Path to the activation scales for converting the model to CTranslate2")
@click.option("--gpt_temperature", type=click.FloatRange(0.0),                           help="Value used to modulate the next token probabilities")
@click.option("--gpt_top_k", type=click.IntRange(0),                                     help="Nmber of highest probability vocabulary tokens to keep for top-k-filtering")
@click.option("--gpt_top_p", type=click.FloatRange(0.0),                                 help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation")
@click.option("--gpt_do_sample", is_flag=True, default=None,                             help="Enable decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling")
@click.option("--gpt_dtype", type=str, default="default",                                help="Torch dtype to use for the model (transformers: default, float32, bfloat16; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# Text to speech options
@click.option("--bark_model", type=str, required=True,                                   help="Bark model for text to speech")
@click.option("--bark_semantic_temperature", type=click.FloatRange(0.0),                 help="Temperature for the bark generation (semantic/text)")
@click.option("--bark_coarse_temperature", type=click.FloatRange(0.0),                   help="Temperature for the bark generation (course waveform)")
@click.option("--bark_fine_temperature", type=click.FloatRange(0.0), default=0.5,        help="Temperature for the bark generation (fine waveform)")
@click.option("--bark_use_better_transformer", is_flag=True, default=False,              help="Optimize bark with BetterTransformer (shorter inference time)")
@click.option("--bark_dtype", type=str, default="default",                               help="Torch dtype to use for the model (default, float32, float16)")
@click.option("--bark_cpu_offload", is_flag=True, default=False,                         help="Offload unused models to the cpu for bark text to speech (lower vram usage, longer inference time)")
# Sentence splitting options
@click.option("--wtpsplit_model", type=str, required=True,                               help="Wtpsplit model for sentence splitting")
# Language options
@click.option("--languages", type=parse_comma_list, default=["english"],                 help="Languages to accept as inputs (stt, tts, text generation & sentence splitting models need to be able to work with the languages provided)")
@click.option("--default_language", type=str, default="english",                         help="Fallback language in case the detected language is not provided")
# Translation
@click.option("--translate", is_flag=True, default=False,                                help="Always translate to and from the default language instead of using a multi language model")
@click.option("--opus_model_names_base", type=str,                                       help="String to be formatted with the corresponding language codes (e.g. 'Helsinki-NLP/opus-mt-{}-{}' -> 'Helsinki-NLP/opus-mt-en-de'), needs trantion to be enabled")
@click.option("--opus_ctranslate_dir", type=click.Path(file_okay=False),                 help="Directory where the CTranslate2 conversion of the model is or should be (this activates CTranslate2)")
@click.option("--opus_dtype", type=str, default="default",                               help="Torch dtype to use for the model (transformers: default, float32; Ctranslate2: default, auto, int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, float32, bfloat16)")
# Prompts & Profiles
@click.option("--profiles", type=click.Path(exists=True, dir_okay=False), required=True, help="The json file that houses the information on all social media profiles (path, url, character, title)")
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
            server.profile_pipe[0],
        ),
    )
    processor.start()

    # Wait for models to load
    server.models_ready.wait()

    # Start server
    try:
        print("Running server at http://0.0.0.0:5000")
        server.sio.run(server.app, host="0.0.0.0", port=5000)
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
