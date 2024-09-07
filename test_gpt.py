# filter out deprication warnings
import warnings

warnings.filterwarnings("ignore")

from transformers import pipeline, set_seed
from time import time

prompt = """This is a series of questions and answers. The answers include non-speech symbols, like [laughter], [laughs], [sighs], [music], [gasps], [clears throat], or ... for hesitations. The answers are multiple sentences long.
Q: Hello? Is there anybody out there?
A:"""

print("loading gpt...")
generator = pipeline("text-generation", model="gpt2-medium")

set_seed(int(time()))

# simple generation
text: str = generator(
    prompt,
    max_new_tokens=128,
    temperature=0.7,
    # top_k=50,
    # top_p=1.0,
    num_return_sequences=1,
)[0]["generated_text"]
text = [s.strip() for s in text.replace(prompt, "").split("\n")]

print(text)
