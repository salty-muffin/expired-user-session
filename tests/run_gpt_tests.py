import subprocess

runs = 5
iterations = 10
prompt = "Hello, is anybody out there?"
parameters = [
    {
        "model": "meta-llama/Llama-3.1-8B",
        "language": "english",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "ctranslate_dir": "server/models/ctranslate2/Llama-3.1-8B",
        "dtype": "int8",
    },
    {
        "model": "facebook/opt-1.3b",
        "language": "english",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "float32",
    },
]
temperatures = [1.0, 1.1, 1.2, 1.3, 1.4]

len_tests = len(parameters) * len(temperatures)
print(f"running {len_tests} tests...")

try:
    for p_i, p in enumerate(parameters):
        do_sample = p.pop("do_sample", False)
        for t_i, t in enumerate(temperatures):
            command = [
                "python3",
                "tests/test_gpt.py",
                f"--runs={runs}",
                f"--iterations={iterations}",
                f"--prompt={prompt}",
                f"--temperature={t}",
            ]
            if do_sample:
                command.append("--do_sample")
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
