import os

from server.sentence_splitter import SentenceSplitter
import time

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")


def test(text: str, model_name: str, device: str) -> None:
    splitter = SentenceSplitter(model_name, device)

    start = time.perf_counter()
    split = splitter.split(text)
    print(time.perf_counter() - start)

    print(split)


if __name__ == "__main__":
    test(
        "The wind whispered through the trees, carrying with it the scent of damp earth and fallen leaves. A thin mist clung to the ground, swirling in delicate tendrils around the roots and stones scattered across the path... Somewhere in the distance, the soft call of a bird echoed through the woods, as if it were speaking to the silence itself. The world felt suspended, caught in the quiet breath between night and day, where time seemed to stretch and contract in ways impossible to understand. It was a moment that felt both fleeting and eternal, like the pause before a dream takes shape.",
        "sat-3l-sm",
        "cpu",
    )
