import os
import time
import random

from text_generator import TextGenerator
from sentence_splitter import SentenceSplitter

from prompts import question_prompt, continuation_prompt

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")


def next_response(
    generator: TextGenerator,
    sentence_splitter: SentenceSplitter,
    gpt_temp: float,
    gpt_top_k: int,
    gpt_top_p: float,
    message: str | None = None,
    responses=[],
) -> str:
    if message:
        responses = []

        prompt = question_prompt.format(message)
    else:
        prompt = continuation_prompt.format(" ".join(responses))

    response_lines = (
        generator.generate(
            prompt,
            temperature=gpt_temp,
            top_k=gpt_top_k,
            top_p=gpt_top_p,
            max_new_tokens=128,
            do_sample=True,
        )
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
    iterations: int, message: str, model: str, temp: float, top_k: int, top_p: float
) -> None:
    text_generator = TextGenerator(model)
    sentence_splitter = SentenceSplitter("segment-any-text/sat-3l-sm", "cpu")

    text_generator.set_seed(int(time.time()))

    response, responses = next_response(
        text_generator, sentence_splitter, temp, top_k, top_p, message
    )
    print(f"0: {response}")
    for i in range(1, iterations):
        response, responses = next_response(
            text_generator, sentence_splitter, temp, top_k, top_p, message, responses
        )
        print(f"{i}: {response}")


if __name__ == "__main__":
    try:
        test(20, "Hello, this is a test.", "facebook/opt-1.3b", 1.1, 50, 1.0)
    except KeyboardInterrupt:
        pass
