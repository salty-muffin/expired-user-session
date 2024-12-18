echo "starting EXPIRED USER SESSION..."
python -u app.py \
    --whisper_model=openai/whisper-base.en \
    --whisper_dtype=float16 \
    --gpt_model=facebook/opt-1.3b \
    --gpt_temperature=1.1 \
    --gpt_top_k=50 \
    --gpt_top_p=1.0 \
    --gpt_do_sample \
    --bark_model=suno/bark \
    --bark_semantic_temperature=1.0 \
    --bark_coarse_temperature=0.6 \
    --wtpsplit_model=segment-any-text/sat-3l-sm \
    prompts.yml