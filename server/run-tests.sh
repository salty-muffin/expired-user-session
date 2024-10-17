echo "RUNNING TESTS..."

# Define the array of temperature values
temperatures=(1.0 1.1 1.2 1.3 1.4)

echo "facebook/opt-1.3b"
# Loop through each temperature value and run the command
for temp in "${temperatures[@]}"; do
    echo "$temp"
    python3 server/test_gpt.py \
        --runs=5 \
        --iterations=10 \
        --prompt="Hello, is anybody out there?" \
        --model="facebook/opt-1.3b" \
        --temperature="$temp" \
        --top_k=50 \
        --top_p=1.0 \
        --do_sample
done

echo "meta-llama/Llama-3.2-1B"
# Loop through each temperature value and run the command
for temp in "${temperatures[@]}"; do
    echo "$temp"
    python3 server/test_gpt.py \
        --runs=5 \
        --iterations=10 \
        --prompt="Hello, is anybody out there?" \
        --model="meta-llama/Llama-3.2-1B" \
        --temperature="$temp" \
        --top_k=50 \
        --top_p=1.0 \
        --use_bfloat16 \
        --do_sample
done

echo "meta-llama/Llama-3.2-3B"
# Loop through each temperature value and run the command
for temp in "${temperatures[@]}"; do
    echo "$temp"
    python3 server/test_gpt.py \
        --runs=5 \
        --iterations=10 \
        --prompt="Hello, is anybody out there?" \
        --model="meta-llama/Llama-3.2-3B" \
        --temperature="$temp" \
        --top_k=50 \
        --top_p=1.0 \
        --use_bfloat16 \
        --do_sample
done