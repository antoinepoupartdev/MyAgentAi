"""
LLM utilities.

This module sends transcribed user text to Gemini and returns a short reply.

Configuration:
    Add your Gemini API key below.

Dependencies:
    pip install requests
"""

from __future__ import annotations

import os

import requests


# These values can be overridden in the .env file.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant for a simple learning project. "
    "Reply clearly, briefly, and conversationally. "
    "Always reply in the same language as the user's message. "
    "If the user speaks French, answer in French. If the user speaks English, answer in English. "
    "Do not restart with a generic greeting unless the user is actually greeting you or starting a conversation. "
    "If the user's input looks like a short non-speech event or accidental sound, ask them to repeat themselves naturally."
)


def _require_api_key() -> None:
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_api_key or "YOUR_" in gemini_api_key:
        raise ValueError(
            "Add your GEMINI_API_KEY in the .env file before running the project."
        )


def generate_response(user_text: str) -> str:
    _require_api_key()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    if not user_text.strip():
        raise ValueError("generate_response() received empty user text.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={gemini_api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{SYSTEM_PROMPT}\n\nUser: {user_text}",
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    reply = "".join(text_parts).strip()

    if not reply:
        raise RuntimeError("Gemini returned an empty text response.")

    return reply
