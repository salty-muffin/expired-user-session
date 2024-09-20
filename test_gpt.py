import os
import nltk
import time
import random

from text_generator import TextGenerator

from prompts import question_prompt, continuation_prompt

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")


def next_response(
    generator: TextGenerator,
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

    sentences = nltk.sent_tokenize(response)
    sentence = sentences[random.randint(0, len(sentences) - 1)]
    return sentence, responses


def test(message: str, model: str, temp: float, top_k: int, top_p: float) -> None:
    nltk.download("punkt_tab")
    text_generator = TextGenerator(model)

    text_generator.set_seed(int(time.time()))

    response, responses = next_response(text_generator, temp, top_k, top_p, message)
    print(f"0: {response}")
    for i in range(1, 5):
        response, responses = next_response(
            text_generator, temp, top_k, top_p, message, responses
        )
        print(f"{i}: {response}")


if __name__ == "__main__":
    try:
        test("Is there a hell?", "facebook/opt-1.3b", 1.1, 50, 1.0)
    except KeyboardInterrupt:
        pass
