# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

import subprocess
import torch
from transformers import pipeline, set_seed, GPT2Tokenizer
import ctranslate2


class TextGenerator:
    def __init__(
        self,
        model_name: str,
        dtype="default",
        device: str | None = None,
        device_map: str | None = None,
    ) -> None:
        if device is None and device_map is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device_map is not None:
            device = None

        dtype_map = {
            "default": None,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map.keys():
            raise ValueError(
                f"dtype for {type(self).__name__} (transformers) only accepts {dtype_map.keys()}"
            )
        torch_dtype = dtype_map[dtype] if "cuda" in device else dtype_map["default"]

        print(f"Using {device} with {dtype} for text generation with '{model_name}'.")

        self._generator = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
            device_map=device_map,
        )

    def set_seed(self, seed: int) -> None:
        set_seed(seed)

    def generate(self, prompt: str, **kwargs) -> str:
        return self._generator(
            prompt, pad_token_id=self._generator.tokenizer.eos_token_id, **kwargs
        )[0]["generated_text"].strip()


class TextGeneratorCTranslate(TextGenerator):
    def __init__(
        self,
        model_name: str,
        ctranslate_dir: str,
        activation_scales: str | None = None,
        dtype="default",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(
            f"Using {device} with {dtype} for text generation (CTranslate2) with '{model_name}'."
        )

        # convert model, if it hasn't been converted yet
        if dtype not in ["default", "auto"]:
            ctranslate_dir += f"--{dtype}"
        command = [
            "ct2-transformers-converter",
            "--model",
            model_name,
            "--output_dir",
            ctranslate_dir,
        ]
        if dtype not in ["default", "auto"]:
            command += ["--quantization", dtype]
        if activation_scales:
            command += ["--activation_scales", activation_scales]
        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        self._generator = ctranslate2.Generator(ctranslate_dir, device=device)

    def generate(self, prompt: str, **kwargs) -> str:
        # set ctranslate2 kwargs
        ct_kwargs = {}
        if "max_length" in kwargs.keys():
            ct_kwargs["max_length"] = kwargs["max_length"]
        elif "max_new_tokens" in kwargs.keys():
            ct_kwargs["max_length"] = len(prompt) + kwargs["max_new_tokens"]
        if "temperature" in kwargs.keys():
            ct_kwargs["sampling_temperature"] = kwargs["temperature"]
        if "top_k" in kwargs.keys():
            ct_kwargs["sampling_topk"] = kwargs["top_k"]
        if "top_p" in kwargs.keys():
            ct_kwargs["sampling_topp"] = kwargs["top_p"]
        if "do_sample" in kwargs.keys():
            ct_kwargs["beam_size"] = 1

        start_tokens = self._tokenizer.convert_ids_to_tokens(
            self._tokenizer.encode(prompt)
        )

        results = self._generator.generate_batch([start_tokens], **ct_kwargs)

        return self._tokenizer.decode(results[0].sequences_ids[0])
