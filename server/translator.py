# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import torch
import itertools
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
)


class Opus:
    def __init__(
        self,
        model_names_base: str,
        languages: list[str],
        mappings={"german": "de", "english": "en"},
        device: str | None = None,
    ) -> None:
        """
        :param model_names_base: A string to be formatted with the corresponding language codes (e.g. "Helsinki-NLP/opus-mt-{}-{}" -> "Helsinki-NLP/opus-mt-en-de")
        :param languages: A list of strings with the languages that can be translated (e.g. ["german", "english"])
        """

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using {device} for Opus text translation using single models.")

        # generate all all possible language pairs
        language_pairs = list(
            itertools.combinations([mappings[lang] for lang in languages], 2)
        )
        self._language_pairs = []
        for pair in language_pairs:
            self._language_pairs.append(pair)
            self._language_pairs.append(tuple(reversed(pair)))

        # load all models
        self._pipes = {}
        for pair in self._language_pairs:
            self._pipes["{}-{}".format(*pair)] = pipeline(
                "translation", model=model_names_base.format(*pair), device=device
            )

        self._mappings = mappings

    def translate(
        self, text: str, source_lang: str, target_lang: str, max_length=256
    ) -> str:
        """
        Translates text with Opus Multilang.

        Returns the translated text.
        """

        return self._pipes[
            "{}-{}".format(self._mappings[source_lang], self._mappings[target_lang])
        ](text, max_length=max_length)[0]["translation_text"]


class T5:
    def __init__(
        self,
        model_name: str,
        task_prefix="translate {source_lang} to {target_lang}: ",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        print(f"Using {device} for T5 text translation.")

        self._tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self._model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self._device
        )

        self._task_prefix = task_prefix

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text with T5.

        Returns the translated text.
        """

        inputs = self._tokenizer(
            self._task_prefix.format_map(
                {"source_lang": source_lang.title(), "target_lang": target_lang.title()}
            )
            + text,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model.generate(inputs.input_ids)

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)


class OpusMul:
    def __init__(
        self,
        model_name: str,
        mappings={"german": "deu", "english": "eng"},
        prefix=">>{}<< ",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(
            f"Using {device} for Opus text translation using the multilang to unknown method."
        )

        self._translator = pipeline("translation", model=model_name, device=device)

        self._mappings = mappings
        self._prefix = prefix

    def translate(self, text: str, lang: str, max_length=256) -> str:
        """
        Translates text with Opus Multilang.

        Returns the translated text.
        """

        return self._translator(
            self._prefix.format(self._mappings[lang]) + text, max_length=max_length
        )[0]["translation_text"]


class OpusSingle:
    def __init__(
        self,
        model_name,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(
            f"Using {device} for Opus text translation using the multilang to unknown method."
        )

        self._translator = pipeline(
            "translation",
            model=model_name,
        )

    def translate(self, text: str, max_length=256) -> str:
        """
        Translates text with Opus Multilang.

        Returns the translated text.
        """

        return self._translator(text, max_length=max_length)[0]["translation_text"]
