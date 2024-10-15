import os
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


def test(iterations: int, message: str, model: str, **kwargs) -> None:
    if huggingface_token := os.environ.get("HUGGINGFACE_TOKEN"):
        login(huggingface_token)

    # get prompts
    with open("server/prompts.yml") as file:
        prompts = yaml.safe_load(file)

    text_generator = TextGenerator(model)
    sentence_splitter = SentenceSplitter("segment-any-text/sat-3l-sm", "cpu")

    text_generator.set_seed(int(time.time()))

    response, responses = next_response(
        text_generator, sentence_splitter, prompts, message, **kwargs
    )
    print(f"0: {response}")
    for i in range(1, iterations):
        response, responses = next_response(
            text_generator,
            sentence_splitter,
            prompts,
            message,
            responses,
            **kwargs,
        )
        print(f"{i}: {response}")


if __name__ == "__main__":
    try:
        test(
            iterations=20,
            message="Hello, this is a test.",
            model="facebook/opt-1.3b",
            temperature=1.1,
            top_k=50,
            top_p=1.0,
            do_sample=True,
        )
    except KeyboardInterrupt:
        pass
