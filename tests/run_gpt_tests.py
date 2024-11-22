import subprocess

runs = 5
iterations = 10

parameters = [
    # --- "Hello, is anybody out there?"
    # meta-llama/Llama-3.1-8B english "Hello, is anybody out there?"
    {
        "prompt": "Hello, is anybody out there?",
        "model": "meta-llama/Llama-3.1-8B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "ctranslate_dir": "server/models/ctranslate2/Llama-3.1-8B",
        "dtype": "int8",
        "language": "english",
        "default_language": "english",
        "translate": False,
    },
    # facebook/opt-1.3b english "Hello, is anybody out there?"
    {
        "prompt": "Hello, is anybody out there?",
        "model": "facebook/opt-1.3b",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "float32",
        "language": "english",
        "default_language": "english",
        "translate": False,
    },
    # meta-llama/Llama-3.1-8B multilang german "Hallo, ist da jemand?"
    {
        "prompt": "Hallo, ist da jemand?",
        "model": "meta-llama/Llama-3.1-8B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "ctranslate_dir": "server/models/ctranslate2/Llama-3.1-8B",
        "dtype": "int8",
        "language": "german",
        "default_language": "english",
        "translate": False,
    },
    # facebook/opt-1.3b translated german "Hallo, ist da jemand?"
    {
        "prompt": "Hallo, ist da jemand?",
        "model": "facebook/opt-1.3b",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "float32",
        "language": "german",
        "default_language": "english",
        "translate": True,
        "opus_model_names_base": "Helsinki-NLP/opus-mt-{}-{}",
    },
    # --- "Are you still mad at me Peter?"
    # meta-llama/Llama-3.1-8B english "Are you still mad at me Peter?"
    {
        "prompt": "Are you still mad at me Peter?",
        "model": "meta-llama/Llama-3.1-8B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "ctranslate_dir": "server/models/ctranslate2/Llama-3.1-8B",
        "dtype": "int8",
        "language": "english",
        "default_language": "english",
        "translate": False,
    },
    # facebook/opt-1.3b english "Are you still mad at me Peter?"
    {
        "prompt": "Are you still mad at me Peter?",
        "model": "facebook/opt-1.3b",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "float32",
        "language": "english",
        "default_language": "english",
        "translate": False,
    },
    # meta-llama/Llama-3.1-8B multilang german "Bist du noch sauer auf mich, Peter?"
    {
        "prompt": "Bist du noch sauer auf mich, Peter?",
        "model": "meta-llama/Llama-3.1-8B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "ctranslate_dir": "server/models/ctranslate2/Llama-3.1-8B",
        "dtype": "int8",
        "language": "german",
        "default_language": "english",
        "translate": False,
    },
    # facebook/opt-1.3b translated german "Bist du noch sauer auf mich, Peter?"
    {
        "prompt": "Bist du noch sauer auf mich, Peter?",
        "model": "facebook/opt-1.3b",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "float32",
        "language": "german",
        "default_language": "english",
        "translate": True,
        "opus_model_names_base": "Helsinki-NLP/opus-mt-{}-{}",
    },
    # --- "Can you describe how it is on the other side?"
    # meta-llama/Llama-3.1-8B english "Can you describe how it is on the other side?"
    {
        "prompt": "Can you describe how it is on the other side?",
        "model": "meta-llama/Llama-3.1-8B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "ctranslate_dir": "server/models/ctranslate2/Llama-3.1-8B",
        "dtype": "int8",
        "language": "english",
        "default_language": "english",
        "translate": False,
    },
    # facebook/opt-1.3b english "Can you describe how it is on the other side?"
    {
        "prompt": "Can you describe how it is on the other side?",
        "model": "facebook/opt-1.3b",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "float32",
        "language": "english",
        "default_language": "english",
        "translate": False,
    },
    # meta-llama/Llama-3.1-8B multilang german "Kannst du beschreiben, wie es ist auf der anderen Seite?"
    {
        "prompt": "Kannst du beschreiben, wie es ist auf der anderen Seite?",
        "model": "meta-llama/Llama-3.1-8B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "ctranslate_dir": "server/models/ctranslate2/Llama-3.1-8B",
        "dtype": "int8",
        "language": "german",
        "default_language": "english",
        "translate": False,
    },
    # facebook/opt-1.3b translated german "Kannst du beschreiben, wie es ist auf der anderen Seite?"
    {
        "prompt": "Kannst du beschreiben, wie es ist auf der anderen Seite?",
        "model": "facebook/opt-1.3b",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "float32",
        "language": "german",
        "default_language": "english",
        "translate": True,
        "opus_model_names_base": "Helsinki-NLP/opus-mt-{}-{}",
    },
]
temperatures = [1.0, 1.1, 1.3]
# temperatures = [1.0, 1.1, 1.2, 1.3, 1.4]

len_tests = len(parameters) * len(temperatures)
print(f"running {len_tests} tests...")

try:
    for p_i, p in enumerate(parameters):
        do_sample = p.pop("do_sample", False)
        translate = p.pop("translate", False)
        for t_i, t in enumerate(temperatures):
            command = [
                "python3",
                "tests/test_gpt.py",
                f"--runs={runs}",
                f"--iterations={iterations}",
                f"--temperature={t}",
            ]
            if do_sample:
                command.append("--do_sample")
            if translate:
                command.append("--translate")
            for key, value in p.items():
                command.append(f"--{key}={value}")

            print(
                f"executing {p_i * len(temperatures) + t_i + 1}/{len_tests}: {' '.join(command)}"
            )
            subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
except KeyboardInterrupt:
    print("exiting...")
