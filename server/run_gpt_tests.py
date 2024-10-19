import subprocess

runs = 5
iterations = 10
prompt = "Hello, is anybody out there?"
parameters = [
    {
        "model": "facebook/opt-1.3b",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "use_bfloat16": False,
    },
    {
        "model": "meta-llama/Llama-3.2-1B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "use_bfloat16": True,
    },
    {
        "model": "meta-llama/Llama-3.2-1B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "use_bfloat16": False,
    },
    {
        "model": "meta-llama/Llama-3.2-3B",
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "use_bfloat16": True,
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
                f"--model={p['model']}",
                f"--top_k={p['top_k']}",
                f"--top_p={p['top_p']}",
                f"--temperature={t}",
            ]
            if p["do_sample"]:
                command.append("--do_sample")
            if p["use_bfloat16"]:
                command.append("--use_bfloat16")

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
