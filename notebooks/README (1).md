# Essay Scorer (Streamlit prototype)

1. Copy from Google Drive (MyDrive/essay-scoring/models/) into ./models/:
   - llama3_aes_lora/      (LoRA weights + tokenizer)
   - fast_heads.joblib     (classifiers + prompt text)
2. pip install -r requirements.txt   (needs an NVIDIA GPU, ~6 GB VRAM, Linux or WSL)
3. streamlit run app.py

No GPU locally? Run it on Colab/Kaggle and expose it with cloudflared or ngrok.
