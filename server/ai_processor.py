import os
import random
import yaml
from collections import Counter

from huggingface_hub import login

from typing import Optional, Any
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event


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
