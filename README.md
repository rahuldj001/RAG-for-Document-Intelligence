# RAG for Document Intelligence

Free-tier RAG pipeline: PDFs are loaded with LangChain `PyPDFLoader`, split with `RecursiveCharacterTextSplitter`, embedded locally with HuggingFace `all-MiniLM-L6-v2`, stored in Supabase pgvector, retrieved with direct `supabase-py` RPC plus BM25 hybrid search, and answered by Groq `llama-3.1-8b-instant`.

## One-time Supabase setup (run the SQL migration, enable pgvector)

1. Create a free Supabase project.
2. Open Supabase Dashboard -> SQL Editor -> New query.
3. Paste the full contents of `supabase/migrations/001_setup.sql`.
4. Click Run. The migration enables `vector`, creates `documents`, creates the `ivfflat` index with `lists=100`, creates `match_documents`, and creates `eval_results`.
5. Go to Project Settings -> API and copy:
   - Project URL as `SUPABASE_URL`
   - `service_role` key as `SUPABASE_SERVICE_KEY`

Use the service role key only on the backend. Do not put it in Streamlit secrets.

## Local development setup (env vars, pip install, run API + Streamlit)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install backend dependencies:

```bash

```

3. Copy `.env.example` to `.env` and fill in:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
GROQ_API_KEY=your-groq-api-key
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.1-8b-instant
CHUNK_SIZE=512
CHUNK_OVERLAP=51
RETRIEVAL_TOP_K=5
```

4. Load env vars and start FastAPI:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

5. In another terminal, start Streamlit:

```bash
pip install -r requirements.streamlit.txt
streamlit run app.py
```

For local Streamlit, it defaults to `http://localhost:8000`. In Streamlit Cloud, set `API_URL` in secrets.

## Deploy backend to Railway (exact clicks + env vars to paste)

1. Push this `doc-intelligence` folder to a GitHub repository.
2. Open Railway -> New Project -> Deploy from GitHub repo.
3. Select the repo and set the root directory to `doc-intelligence` if the repo contains other folders.
4. Railway detects Python through `requirements.txt`. The included `Procfile` and `railway.toml` start:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

5. Open the Railway service -> Variables -> New Variable and paste:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
GROQ_API_KEY=your-groq-api-key
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.1-8b-instant
CHUNK_SIZE=512
CHUNK_OVERLAP=51
RETRIEVAL_TOP_K=5
```

6. Open Settings -> Networking -> Generate Domain.
7. Verify the backend at:

```bash
https://your-railway-app.up.railway.app/health
```

Railway free tier sleeps after inactivity. The Streamlit app calls `/health` on load as a keep-alive ping.

## Deploy UI to Streamlit Community Cloud (exact steps)

1. Open Streamlit Community Cloud -> New app.
2. Select the same GitHub repo.
3. Set the branch.
4. Set the main file path to `doc-intelligence/app.py`.
5. In Advanced settings, set the requirements file to `doc-intelligence/requirements.streamlit.txt` if prompted.
6. Add secrets using the structure from `secrets.toml.example`:

```toml
API_URL = "https://your-railway-app.up.railway.app"
```

7. Click Deploy.

The UI only needs `streamlit` and `requests`; all Supabase, embedding, LangChain, and Groq dependencies stay on Railway.

## How to ingest a PDF and verify it's stored in Supabase

1. Open the Streamlit app.
2. Use the sidebar PDF uploader.
3. Leave `Force reingest` unchecked for normal deduplication.
4. Click Ingest.
5. In Supabase, open Table Editor -> `documents`.
6. Verify rows exist with:
   - `content`
   - `embedding`
   - `metadata.filename`
   - `metadata.page`
   - `metadata.chunk_index`
   - `metadata.file_hash`

Deduplication computes the PDF MD5 hash and skips ingestion when chunks with the same `metadata.file_hash` already exist. Set `force_reingest=true` to delete and replace chunks for that exact hash.

## How to run RAGAS evaluation and read the scores

1. In Streamlit, open the Evaluate expander and click Run RAGAS evaluation.
2. Or call the backend:

```bash
curl -X POST https://your-railway-app.up.railway.app/evaluate \
  -H "Content-Type: application/json" \
  -d "{\"run_name\":\"manual-eval\"}"
```

3. Open Supabase -> Table Editor -> `eval_results`.
4. Read:
   - `scores`: current `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`
   - `deltas`: change versus the previous stored run
   - `sample_count`: number of evaluation questions

You can pass curated samples:

```json
{
  "run_name": "curated-v1",
  "samples": [
    {
      "question": "What is the refund policy?",
      "ground_truth": "Refunds are available within 30 days."
    }
  ]
}
```

## Free tier limits table — what runs out first and how to stay within limits

| Service | Free tier limit | What runs out first | How to stay within limits |
| --- | --- | --- | --- |
| Supabase | 500MB database storage | Embeddings and chunk text | Keep `CHUNK_SIZE=512`, delete old PDFs, avoid duplicate ingestion, store only needed metadata. |
| Groq | 14,400 requests/day | Chat and evaluation judge calls | Cache repeated answers if needed, keep `RETRIEVAL_TOP_K=5`, run evals on small curated sets. |
| HuggingFace local embeddings | Free on your compute | Railway RAM/CPU during model load | Use `all-MiniLM-L6-v2`, ingest modest PDFs, batch inserts at 100 rows. |
| Railway | 500 hours/month | Backend uptime and cold starts | Let the service sleep, rely on Streamlit `/health` ping when users open the app. |
| Streamlit Community Cloud | Free app hosting | App resource limits | Keep UI dependencies scoped to `requirements.streamlit.txt`, run embeddings only on backend. |
