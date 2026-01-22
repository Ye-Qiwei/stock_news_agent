# Stock News Agents

A LangChain + LangGraph multi-agent demo that charts weekly stock prices and retrieves news for a selected week, producing Chinese summaries with sentiment scores.

## Features
- Streamlit UI: ticker/company input, US/JP markets, and 3m/1y/5y time ranges.
- Price data from Stooq via the MCP price server.
- Click a weekly price point to fetch company and industry news for that week.
- News from Google News RSS via the MCP news server (zh/ja/en sources).
- LLM-generated news summaries (3 Chinese sentences + sentiment score).
- RAG (Chroma) cache with a UI button to clear stored news.

## Install
```bash
pip install -e .
```

## Environment variables
### LLM (chat/summary)
Default (OpenAI):
```bash
export STOCK_CHAT_PROVIDER=openai
export OPENAI_API_KEY=YOUR_KEY
export STOCK_CHAT_MODEL=gpt-4o-mini
export STOCK_SUMMARY_MODEL=gpt-4o-mini
export STOCK_SUMMARY_CONCURRENCY=6
```
Optional: set a separate provider for summaries (defaults to `STOCK_CHAT_PROVIDER`):
```bash
export STOCK_SUMMARY_PROVIDER=openai
```

Switch to Groq:
```bash
export STOCK_CHAT_PROVIDER=groq
export GROQ_API_KEY=YOUR_GROQ_KEY
export STOCK_CHAT_MODEL=llama3-8b-8192
export STOCK_SUMMARY_MODEL=llama3-8b-8192
```
Switch to xAI Grok:
```bash
export STOCK_CHAT_PROVIDER=xai
export XAI_API_KEY=YOUR_XAI_KEY
export STOCK_CHAT_MODEL=grok-2-latest
```
Switch to Qwen (DashScope-compatible API):
```bash
export STOCK_CHAT_PROVIDER=qwen
export DASHSCOPE_API_KEY=YOUR_DASHSCOPE_KEY
export STOCK_CHAT_MODEL=qwen-plus
```
Switch to local (Ollama):
```bash
export STOCK_CHAT_PROVIDER=ollama
export STOCK_CHAT_MODEL=qwen2.5:7b-instruct
```

### Embeddings (RAG)
Default (OpenAI embeddings):
```bash
export STOCK_EMBED_PROVIDER=openai
export STOCK_EMBED_MODEL=text-embedding-3-small
```
Switch to Jina embeddings:
```bash
export STOCK_EMBED_PROVIDER=jina
export STOCK_EMBED_API_KEY=YOUR_JINA_KEY
export STOCK_EMBED_MODEL=jina-embeddings-v3
```

### Data + thresholds
```bash
export STOCK_DATA_DIR=/path/to/data
export STOCK_NEWS_RAG_THRESHOLD=0.62
```
- RAG data is stored at `${STOCK_DATA_DIR}/rag_store`.

### News MCP (optional)
The news server loads `OPENAI_API_KEY` from a local `.env` file to expand Google News queries. If unset, query expansion is skipped.

## Run
```bash
streamlit run src/stock_agents/app.py
```

## API test commands
OpenAI:
```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}'
```
Groq:
```bash
curl https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-8192","messages":[{"role":"user","content":"ping"}]}'
```
Grok:
```bash
curl https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-2-latest","messages":[{"role":"user","content":"ping"}]}'
```
Qwen:
```bash
curl https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-plus","messages":[{"role":"user","content":"ping"}]}'
```
Jina embeddings:
```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"jina-embeddings-v3","input":["ping"]}'
```

## Notes
- Price data comes from Stooq CSV; the MCP server handles missing headers and date/close cleaning.
- The chart uses weekly resampling to avoid a straight-line plot for dense data.
- Company news queries prefer ticker + company name, while industry news uses an LLM-inferred industry label.
