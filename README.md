# Medical-Agentic-Chatbot

Commands to Execute:
1. conda create -n project python=3.11 -y
2. conda activate project
3. pip install -r requirements.txt
4. add  .env

# Edit .env with real API keys

python ingest.py --pdf data/medical_book.pdf

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

