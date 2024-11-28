import os
import re
import click
import time
import random
from dotenv import load_dotenv
import yaml

from huggingface_hub import login

import sys

sys.path.append("server/")

from text_generator import TextGenerator, TextGeneratorCTranslate, TextGeneratorAirLLM
from sentence_splitter import SentenceSplitter
from translator import Opus, OpusCTranslate2

load_dotenv()
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "server", "models")


def next_response(
    generator: TextGenerator,
    sentence_splitter: SentenceSplitter,
    language: str,
    default_language: str,
    translate: bool,
    translator: Opus,
    prompts: dict[str, str],
    message: str | None = None,
    responses=[],
    **kwargs,
) -> str:
    lang = default_language if translate else language

    if message:
        if translate and translator and language != default_language:
            message = translator.translate(message, language, default_language)

        responses = []

        prompt = prompts[lang]["question_prompt"].format(message)
    else:
        prompt = prompts[lang]["continuation_prompt"].format(" ".join(responses))

    generator_kwargs = {
        key: value for key, value in kwargs.items() if value is not None
    }
    response_lines = generator.generate(prompt, max_new_tokens=64, **generator_kwargs)[
        len(prompt) : :
    ].split("\n")
    response_lines = [line.strip() for line in response_lines if line]
    if not len(response_lines):
        return "..."
    response = response_lines[0]

    responses.append(response)

    sentences = sentence_splitter.split(response)
    sentence = sentences[random.randint(0, len(sentences) - 1)]

    if translate and translator and language != default_language:
        sentence = translator.translate(sentence, default_language, language)
    return sentence, responses


def test(
    iterations: int,
    message: str,
    language: str,
    default_language: str,
    translate: bool,
    opus_model_names_base: str | None,
    opus_ctranslate_dir: str | None,
    model: str,
    ctranslate_dir: str | None,
    activation_scales: str | None,
    airllm: bool = False,
    compression: str | None = None,
    device_map: str | None = None,
    dtype: str = "default",
    print_file=None,
    **kwargs,
) -> None:
    if huggingface_token := os.environ.get("HF_TOKEN"):
        login(huggingface_token)

    # Get prompts
    with open("server/prompts.yml") as file:
        prompts = yaml.safe_load(file)

    if ctranslate_dir:
        text_generator = TextGeneratorCTranslate(
            model,
            ctranslate_dir=ctranslate_dir,
            activation_scales=activation_scales,
            dtype=dtype,
        )
    elif airllm:
        text_generator = TextGeneratorAirLLM(model, compression=compression)
    else:
        text_generator = TextGenerator(
            model,
            dtype=dtype,
            device_map=device_map,
        )
    sentence_splitter = SentenceSplitter("segment-any-text/sat-3l-sm", "cpu")

    translator = None
    if translate and opus_model_names_base:
        if opus_ctranslate_dir:
            translator = OpusCTranslate2(
                opus_model_names_base,
                [language, default_language],
                ctranslate_dir=opus_ctranslate_dir,
                device="cpu",
            )
        else:
            translator = Opus(
                opus_model_names_base,
                [language, default_language],
                device="cpu",
            )

    text_generator.set_seed(int(time.time()))

    response, responses = next_response(
        text_generator,
        sentence_splitter,
        language,
        default_language,
        translate,
        translator,
        prompts,
        message,
        **kwargs,
    )
    print(f"0: {response}", file=print_file, flush=True)
    for i in range(1, iterations):
        response, responses = next_response(
            text_generator,
            sentence_splitter,
            language,
            default_language,
            translate,
            translator,
            prompts,
            message,
            responses,
            **kwargs,
        )
        print(f"{i}: {response}", file=print_file, flush=True)


def format_prompt_for_filename(s: str) -> str:
    return re.sub(r"""[,!?."']""", "", s).strip().replace(" ", "-")


# fmt: off
@click.command
@click.option("--runs", type=int, required=True)
@click.option("--iterations", type=int, required=True)
@click.option("--prompt", type=str, required=True)
@click.option("--language", type=str, default="english", required=True)
@click.option("--default_language", type=str, default="english")
@click.option("--translate", is_flag=True, default=False)
@click.option("--opus_model_names_base", type=str)
@click.option("--opus_ctranslate_dir", type=str)
@click.option("--model", type=str, required=True)
@click.option("--device", type=str, default=None)
@click.option("--device_map", type=str, default=None)
@click.option("--ctranslate_dir", type=click.Path(file_okay=False))
@click.option("--activation_scales", type=click.Path(exists=True, dir_okay=False))
@click.option("--airllm", is_flag=True, default=False)
@click.option("--compression", type=str, default=None)
@click.option("--temperature", type=float, default=1.0)
@click.option("--top_k", type=int, default=50)
@click.option("--top_p", type=float, default=1.0)
@click.option("--do_sample", is_flag=True, default=False)
@click.option("--dtype", type=str, default="default")
# fmt: on
def run_test(
    runs: int,
    iterations: int,
    prompt: str,
    language: str,
    default_language: str,
    translate: bool,
    opus_model_names_base: str | None,
    opus_ctranslate_dir: str | None,
    model: str,
    device: str,
    device_map: str,
    ctranslate_dir: str,
    activation_scales: str,
    airllm: bool,
    compression: str,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    dtype: bool,
):
    try:
        with open(
            os.path.join(
                "server",
                "tests",
                f"{format_prompt_for_filename(prompt)}_{model.split('/')[-1]}_temp{temperature}{f'_top_k{top_k}_top_p{top_p}' if do_sample else ''}_{dtype}_{language}{'_translated' if translate else ''}.txt",
            ),
            "w+",
        ) as file:
            print(f"executing {runs} runs.", file=file, flush=True)
            print(f"prompt:\t\t\t\t{prompt}", file=file, flush=True)
            print(f"language:\t\t\t{language}", file=file, flush=True)
            print(f"default language:\t{default_language}", file=file, flush=True)
            print(f"translate:\t\t\t{translate}", file=file, flush=True)
            print(f"model:\t\t\t\t{model}", file=file, flush=True)
            print(f"ctranslate:\t\t\t{bool(ctranslate_dir)}", file=file, flush=True)
            print(f"device_map:\t\t\t{device_map}", file=file, flush=True)
            print(f"temperature:\t\t{temperature}", file=file, flush=True)
            print(f"top_k:\t\t\t\t{top_k}", file=file, flush=True)
            print(f"top_p:\t\t\t\t{top_p}", file=file, flush=True)
            print(f"sample:\t\t\t\t{do_sample}", file=file, flush=True)
            print(f"dtype:\t\t\t\t{dtype}", file=file, flush=True)
            print("", file=file)

            for _ in range(0, runs):
                test(
                    iterations=iterations,
                    message=prompt,
                    language=language,
                    default_language=default_language,
                    translate=translate,
                    opus_model_names_base=opus_model_names_base,
                    opus_ctranslate_dir=opus_ctranslate_dir,
                    model=model,
                    ctranslate_dir=ctranslate_dir,
                    activation_scales=activation_scales,
                    airllm=airllm,
                    compression=compression,
                    device=device,
                    device_map=device_map,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    dtype=dtype,
                    print_file=file,
                )
                print("", file=file, flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_test()
