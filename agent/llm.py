"""Chat model selection behind one LangChain interface.

  bedrock    ChatBedrockConverse, model from BEDROCK_MODEL_ID (chosen by check_access.py)
  anthropic  ChatAnthropic, key from LLM_API_KEY or Secrets Manager (LLM_API_KEY_SECRET_NAME)
  openai     ChatOpenAI, same key handling
  none       no LLM: deterministic answers built from the same tool calls (local mode/tests)

If Bedrock refuses a call (access denied, model not enabled) and a fallback key is
configured, the agent switches to LLM_FALLBACK_PROVIDER (default anthropic) for the rest
of the Lambda's life.
"""
from __future__ import annotations

import os
from functools import lru_cache

from traffic_common import store
from traffic_common.config import Settings

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}


@lru_cache(maxsize=1)
def api_key() -> str | None:
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    name = os.environ.get("LLM_API_KEY_SECRET_NAME")
    if not name:
        return None
    try:
        return store.client("secretsmanager").get_secret_value(SecretId=name)["SecretString"].strip()
    except Exception as exc:  # noqa: BLE001
        print(f"cannot read secret {name}: {exc}")
        return None


def make_model(provider: str, st: Settings):
    model = os.environ.get("LLM_MODEL") or None
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse
        mid = st.bedrock_model_id or os.environ.get("BEDROCK_MODEL_ID")
        if not mid:
            raise RuntimeError("BEDROCK_MODEL_ID not set")
        return ChatBedrockConverse(model=mid, region_name=st.region, temperature=0.2, max_tokens=900)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or DEFAULT_MODELS["anthropic"], api_key=api_key(), temperature=0.2,
                             max_tokens=900, timeout=25)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or DEFAULT_MODELS["openai"], api_key=api_key(), temperature=0.2,
                          max_tokens=900, timeout=25)
    return None


class LLM:
    """Holds the current provider and swaps to the fallback when Bedrock refuses."""

    def __init__(self, st: Settings | None = None):
        self.st = st or Settings.from_env()
        self.provider = self.st.llm_provider
        self.fallback = os.environ.get("LLM_FALLBACK_PROVIDER", "anthropic")
        self.model = None if self.provider == "none" else self._safe_make(self.provider)

    def _safe_make(self, provider):
        try:
            return make_model(provider, self.st)
        except Exception as exc:  # noqa: BLE001
            print(f"LLM provider {provider} unavailable: {exc}")
            return self._fallback_model(exc)

    def _fallback_model(self, exc):
        if self.provider == "bedrock" and api_key():
            print(f"falling back to {self.fallback} ({exc})")
            self.provider = self.fallback
            return make_model(self.fallback, self.st)
        self.provider = "none"
        return None

    @property
    def available(self) -> bool:
        return self.model is not None

    def invoke(self, messages, tools=None):
        model = self.model.bind_tools(tools) if tools else self.model
        try:
            return model.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            if self.provider != "bedrock" or not _is_access_error(exc):
                raise
            self.model = self._fallback_model(exc)
            if not self.model:
                raise
            return self.invoke(messages, tools)


def _is_access_error(exc) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(k in text for k in ("AccessDenied", "not authorized", "ValidationException", "ResourceNotFound",
                                   "don't have access", "UnrecognizedClient"))


def text_of(msg) -> str:
    t = getattr(msg, "text", None)          # property (str subclass) in langchain-core >= 1
    if isinstance(t, str) and t:
        return str(t)
    if callable(t):
        t = t()
        if isinstance(t, str) and t:
            return t
    c = msg.content
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)
