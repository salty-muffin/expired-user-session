import subprocess

runs = 5
iterations = 10
prompt = "Hello, is anybody out there?"
parameters = [
    {
        "model": "facebook/opt-1.3b",
        "language": "english",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "bfloat32",
    },
    {
        "model": "meta-llama/Llama-3.2-1B",
        "language": "english",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "bfloat16",
    },
    {
        "model": "meta-llama/Llama-3.2-1B",
        "language": "english",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "bfloat32",
    },
    {
        "model": "meta-llama/Llama-3.2-3B",
        "language": "english",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "dtype": "bfloat16",
    },
]
temperatures = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

len_tests = len(parameters) * len(temperatures)
print(f"running {len_tests} tests...")

try:
    for p_i, p in enumerate(parameters):
        for t_i, t in enumerate(temperatures):
            command = [
                "python3",
                "server/test_gpt.py",
                f"--runs={runs}",
                f"--iterations={iterations}",
                f"--prompt={prompt}",
                f"--temperature={t}",
            ]
            if p.pop("do_sample"):
                command.append("--do_sample")
            for key, value in p.items():
                command.append(f"--{key}={value}")

            print(
                f"executing {p_i * len(temperatures) + t_i + 1}/{len_tests}: {' '.join(command)}"
            )
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
except KeyboardInterrupt:
    print("exiting...")
