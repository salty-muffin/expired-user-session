import os
from server.translator import T5, OpusMul, Opus

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")


def test_sentence_T5(text):
    translator = T5("google-t5/t5-base")

    translation = translator.translate(
        text, source_lang="english", target_lang="german"
    )

    print(f"translation: '{translation}'")


def test_file_T5(path):
    translator = T5("google-t5/t5-large", device="cpu")

    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        print(
            translator.translate(
                line.strip(), source_lang="english", target_lang="german"
            )
            if line.strip()
            else ""
        )


def test_sentence_OpusMul(text):
    translator = OpusMul("Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-mul")

    translation = translator.translate(text, lang="german")

    print(f"translation: '{translation}'")


def test_file_OpusMul(path):
    translator = OpusMul("Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-mul")

    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        print(translator.translate(line.strip(), lang="german") if line.strip() else "")


def test_sentence_Opus(text):
    translator = Opus("Helsinki-NLP/opus-mt-{}-{}", ["english", "german"])

    translation = translator.translate(
        text, source_lang="english", target_lang="german"
    )

    print(f"translation: '{translation}'")


def test_file_Opus(path):
    translator = Opus("Helsinki-NLP/opus-mt-{}-{}", ["english", "german"])

    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        print(
            translator.translate(
                line.strip(), source_lang="english", target_lang="german"
            )
            if line.strip()
            else ""
        )


if __name__ == "__main__":
    test_sentence_Opus("Hello, is anybody out there?")
    test_file_Opus("server/tests/good/opt-1.3b_temp1.0_top_k50_top_p1.0.txt")
