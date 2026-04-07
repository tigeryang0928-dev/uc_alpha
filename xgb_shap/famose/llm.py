"""LLM backends for FAMOSE: OpenAI-compatible chat and Google Gemini (REST, stdlib only)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    base_url: str,
    api_key: str | None = None,
) -> str:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or pass api_key=")
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode("utf-8", errors="replace")) from e
    return str(data["choices"][0]["message"]["content"])


def _gemini_rest_url(model: str, api_key: str) -> str:
    m = model.strip().replace("/", "")
    q = urllib.parse.urlencode({"key": api_key})
    return (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{m}:generateContent?{q}"
    )


def _messages_to_gemini_body(
    messages: list[dict[str, str]], *, temperature: float
) -> dict:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        text = str(m.get("content", ""))
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            user_parts.append(text)
        elif role == "assistant":
            user_parts.append(f"[assistant]\n{text}")
    user_text = "\n\n".join(user_parts).strip() or "."
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": float(temperature)},
    }
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    return body


def _gemini_text_from_response(data: dict) -> str:
    cands = data.get("candidates")
    if not cands:
        fb = data.get("promptFeedback")
        raise RuntimeError(f"Gemini returned no candidates (promptFeedback={fb!r})")
    first = cands[0]
    reason = first.get("finishReason")
    parts = (first.get("content") or {}).get("parts") or []
    texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
    out = "".join(texts).strip()
    if not out and reason and reason not in ("STOP", "MAX_TOKENS"):
        raise RuntimeError(f"Gemini empty response (finishReason={reason!r})")
    if not out:
        raise RuntimeError(f"Gemini empty text (finishReason={reason!r})")
    return out


def gemini_generate_content(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    api_key: str | None = None,
) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass api_key=")
    url = _gemini_rest_url(model, key)
    payload = _messages_to_gemini_body(messages, temperature=temperature)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode("utf-8", errors="replace")) from e
    return _gemini_text_from_response(data)


def famose_chat_completion(
    messages: list[dict[str, str]],
    *,
    provider: str,
    model: str,
    temperature: float,
    base_url: str,
    api_key: str | None = None,
) -> str:
    p = (provider or "openai").strip().lower()
    if p in ("openai", "openai_compatible", "openai-compat"):
        return chat_completion(
            messages,
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
    if p in ("gemini", "google", "google_gemini"):
        return gemini_generate_content(
            messages, model=model, temperature=temperature, api_key=api_key
        )
    raise ValueError(
        f"Unknown famose llm_provider={provider!r}; use openai or gemini"
    )


def parse_proposal_json(text: str) -> dict | None:
    t = text.strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None
