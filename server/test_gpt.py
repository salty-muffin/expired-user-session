from typing import Literal

import os
import click
import time
import random
from dotenv import load_dotenv
import yaml

from huggingface_hub import login

from text_generator import TextGenerator
from sentence_splitter import SentenceSplitter

load_dotenv()
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")


def next_response(
    generator: TextGenerator,
    sentence_splitter: SentenceSplitter,
    prompts: dict[str, str],
    message: str | None = None,
    responses=[],
    **kwargs,
) -> str:
    if message:
        responses = []

        prompt = prompts["question_prompt"].format(message)
    else:
        prompt = prompts["continuation_prompt"].format(" ".join(responses))

    generator_kwargs = {
        key: value for key, value in kwargs.items() if value is not None
    }
    response_lines = (
        generator.generate(prompt, max_new_tokens=128, **generator_kwargs)
        .replace(prompt, "")
        .split("\n")
    )
    response_lines = [line.strip() for line in response_lines if line]
    if not len(response_lines):
        return "..."
    response = response_lines[0]
    responses.append(response)

    sentences = sentence_splitter.split(response)
    sentence = sentences[random.randint(0, len(sentences) - 1)]
    return sentence, responses


def test(
    iterations: int,
    message: str,
    model: str,
    device_map: Literal["auto"] | None = None,
    use_bfloat16: bool = False,
    print_file=None,
    **kwargs,
) -> None:
    if huggingface_token := os.environ.get("HUGGINGFACE_TOKEN"):
        login(huggingface_token)

    # get prompts
    with open("server/prompts.yml") as file:
        prompts = yaml.safe_load(file)

    text_generator = TextGenerator(
        model, device_map=device_map, use_bfloat16=use_bfloat16
    )
    sentence_splitter = SentenceSplitter("segment-any-text/sat-3l-sm", "cpu")

    text_generator.set_seed(int(time.time()))

    response, responses = next_response(
        text_generator, sentence_splitter, prompts, message, **kwargs
    )
    print(f"0: {response}", file=print_file, flush=True)
    for i in range(1, iterations):
        response, responses = next_response(
            text_generator,
            sentence_splitter,
            prompts,
            message,
            responses,
            **kwargs,
        )
        print(f"{i}: {response}", file=print_file, flush=True)


# fmt: off
@click.command
@click.option("--runs", type=int, required=True)
@click.option("--iterations", type=int, required=True)
@click.option("--prompt", type=str, required=True)
@click.option("--model", type=str, required=True)
@click.option("--temperature", type=float, default=1.0)
@click.option("--top_k", type=int, default=50)
@click.option("--top_p", type=float, default=1.0)
@click.option("--do_sample", is_flag=True, default=False)
@click.option("--use_bfloat16", is_flag=True, default=False)
# fmt: on
def run_test(
    runs: int,
    iterations: int,
    prompt: str,
    model: str,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    use_bfloat16: bool,
):
    try:
        with open(
            os.path.join(
                "server",
                "tests",
                f"{model.split('/')[-1]}_temp{temperature}{f'_top_k{top_k}_top_p{top_p}' if do_sample else ''}{'_bfloat16' if use_bfloat16 else ''}.txt",
            ),
            "w+",
        ) as file:
            print(f"executing {runs} runs.", file=file, flush=True)
            print(f"model: \t\t\t{model}", file=file, flush=True)
            print(f"prompt: \t\t{prompt}", file=file, flush=True)
            print(f"temperature: \t{temperature}", file=file, flush=True)
            print(f"top_k: \t\t\t{top_k}", file=file, flush=True)
            print(f"top_p: \t\t\t{top_p}", file=file, flush=True)
            print(f"sample: \t\t{do_sample}", file=file, flush=True)
            print(f"bfloat16: \t\t{use_bfloat16}", file=file, flush=True)
            print("", file=file)

            for _ in range(0, runs):
                test(
                    iterations=iterations,
                    message=prompt,
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    use_bfloat16=use_bfloat16,
                    print_file=file,
                )
                print("", file=file, flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_test()
