echo "starting EXPIRED USER SESSION..."
python -u app.py \
    --whisper_model=openai/whisper-small \
    --whisper_use_float16 \
    --gpt_model=meta-llama/Llama-3.2-1B \
    --gpt_use_bfloat16 \
    --gpt_temperature=1.1 \
    --gpt_top_k=50 \
    --gpt_top_p=1.0 \
    --gpt_do_sample \
    --bark_model=suno/bark \
    --bark_semantic_temperature=1.0 \
    --bark_coarse_temperature=0.6 \
    --wtpsplit_model=segment-any-text/sat-3l-sm \
    --languages=english,german \
    --default_language=english \
    prompts.yml