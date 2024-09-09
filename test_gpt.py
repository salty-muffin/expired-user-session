# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

from text_generator import load_generator, set_generator_seed, generate

from prompts import question_prompt, continuation_prompt

responses = []


def generate_next_response(message: str | None = None) -> str:
    global responses

    if message:
        responses = []

        prompt = question_prompt.format(message)
    else:
        prompt = continuation_prompt.format(" ".join(responses))

    response_lines = (
        generate(
            prompt,
            max_new_tokens=128,
            do_sample=True,
        )
        .replace(prompt, "")
        .split("\n")
    )
    response_lines = [line.strip() for line in response_lines if line]
    response = response_lines[0]
    responses.append(response)
    return response


if __name__ == "__main__":
    load_generator("facebook/opt-1.3b")
    set_generator_seed(42)

    message = "Is anybody out there?"

    try:
        print(generate_next_response(message))

        while True:
            print(generate_next_response())
    except KeyboardInterrupt:
        raise SystemExit
