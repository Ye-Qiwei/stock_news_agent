import os
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI


def get_chat_model(temperature: float = 0) -> Any:
    provider = os.getenv("STOCK_CHAT_PROVIDER", "openai").lower()
    model = os.getenv("STOCK_CHAT_MODEL", "gpt-4o-mini")
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "groq":
        api_key = os.getenv("STOCK_CHAT_API_KEY") or os.getenv("GROQ_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    if provider == "xai":
        api_key = os.getenv("STOCK_CHAT_API_KEY") or os.getenv("XAI_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
    if provider == "qwen":
        api_key = os.getenv("STOCK_CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return ChatOllama(model=model, temperature=temperature)


def get_summary_model(temperature: float = 0) -> Any:
    provider = os.getenv("STOCK_SUMMARY_PROVIDER", os.getenv("STOCK_CHAT_PROVIDER", "openai")).lower()
    model = os.getenv("STOCK_SUMMARY_MODEL", os.getenv("STOCK_CHAT_MODEL", "gpt-4o-mini"))
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "groq":
        api_key = os.getenv("STOCK_CHAT_API_KEY") or os.getenv("GROQ_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    if provider == "xai":
        api_key = os.getenv("STOCK_CHAT_API_KEY") or os.getenv("XAI_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
    if provider == "qwen":
        api_key = os.getenv("STOCK_CHAT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return ChatOllama(model=model, temperature=temperature)
