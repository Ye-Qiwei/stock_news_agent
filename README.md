# Stock News Agents

A minimal LangChain + LangGraph multi-agent app that visualizes stock prices and retrieves multi-language news using MCP and RAG.

## Features
- Price chart with 3m / 1y / 5y ranges and interactive weekly click.
- Two-direction news search: company (ticker only) and industry (LLM-inferred label).
- Multi-language news (zh/ja/en) with media + blog sources.
- MCP servers for price and news.
- RAG store for cached news and a UI button to clear it.

## Setup
1. Create a virtualenv and install dependencies:
   ```bash
   pip install -e .
   ```
2. Set environment variables (default uses OpenAI):
   ```bash
   export STOCK_CHAT_PROVIDER=openai
   export OPENAI_API_KEY=YOUR_KEY
   export STOCK_CHAT_MODEL=gpt-4o-mini
   export STOCK_SUMMARY_MODEL=gpt-4o-mini
   export STOCK_SUMMARY_CONCURRENCY=6
   export STOCK_EMBED_PROVIDER=openai
   export STOCK_EMBED_MODEL=text-embedding-3-small
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
   Switch to Qwen (DashScope compatible API):
   ```bash
   export STOCK_CHAT_PROVIDER=qwen
   export DASHSCOPE_API_KEY=YOUR_DASHSCOPE_KEY
   export STOCK_CHAT_MODEL=qwen-plus
   ```
   Switch to Jina embeddings:
   ```bash
   export STOCK_EMBED_PROVIDER=jina
   export STOCK_EMBED_API_KEY=YOUR_JINA_KEY
   export STOCK_EMBED_MODEL=jina-embeddings-v3
   ```
   Switch to local model (Ollama):
   ```bash
   export STOCK_CHAT_PROVIDER=ollama
   export STOCK_CHAT_MODEL=qwen2.5:7b-instruct
   ```
   ```bash
   export STOCK_CHAT_PROVIDER=ollama
   export STOCK_CHAT_MODEL=qwen2.5:7b-instruct
   ```

## API Test Command
```bash
curl https://api.openai.com/v1/chat/completions \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{\"model\":\"gpt-4o-mini\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}' 
```
Groq API test:
```bash
curl https://api.groq.com/openai/v1/chat/completions \\
  -H "Authorization: Bearer $GROQ_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{\"model\":\"llama3-8b-8192\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}' 
```
Grok API test:
```bash
curl https://api.x.ai/v1/chat/completions \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{\"model\":\"grok-2-latest\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}' 
```
Qwen API test:
```bash
curl https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \\
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{\"model\":\"qwen-plus\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}' 
```
Jina embeddings test:
```bash
curl https://api.jina.ai/v1/embeddings \\
  -H "Authorization: Bearer $JINA_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{\"model\":\"jina-embeddings-v3\",\"input\":[\"ping\"]}'
```

## Run
```bash
streamlit run src/stock_agents/app.py
```

## Notes
- The app uses Stooq CSV for prices. The MCP server handles missing headers.
- The chart uses weekly resampling to avoid a straight-line plot when data is dense.
- Only tickers are used for search queries (no company names).
