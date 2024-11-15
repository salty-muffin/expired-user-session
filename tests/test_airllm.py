from airllm import AutoModel

MAX_LENGTH = 128

model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-8B", profiling_mode=True, compression="4bit"
)

input_text = [
    "Q: Hello is anybody out there?\nA:",
]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    # padding=True
)

generation_output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(generation_output.sequences[0]))
