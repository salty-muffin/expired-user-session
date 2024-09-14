import nltk
import time
import random

from modules.text_generator import load_generator, set_generator_seed, generate

from prompts import question_prompt, continuation_prompt

responses = []


def generate_next_response(
    temp: float, top_k: int, top_p: float, message: str | None = None
) -> str:
    global responses

    if message:
        responses = []

        prompt = question_prompt.format(message)
    else:
        prompt = continuation_prompt.format(" ".join(responses))

    response_lines = (
        generate(
            prompt,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=128,
            do_sample=True,
        )
        .replace(prompt, "")
        .split("\n")
    )
    response_lines = [line.strip() for line in response_lines if line]
    response = response_lines[0]
    responses.append(response)

    sentences = nltk.sent_tokenize(response)
    sentence = sentences[random.randint(0, len(sentences) - 1)]
    return sentence


def test(message: str, model: str, temp: float, top_k: int, top_p: float) -> None:
    load_generator(model)
    nltk.download("punkt_tab")

    set_generator_seed(int(time.time()))

    print(f"0: {generate_next_response(temp, top_k, top_p, message)}")
    for i in range(1, 5):
        print(f"{i}: {generate_next_response(temp, top_k, top_p)}")


if __name__ == "__main__":
    try:
        test("Is there a hell?", "facebook/opt-1.3b", 1.1, 50, 1.0)
    except KeyboardInterrupt:
        pass
