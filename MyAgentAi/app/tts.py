"""
Text-to-speech utilities.

This module:
1. Sends text to a TTS API
2. Plays the returned WAV audio through the speakers

Configuration:
    Update the .env file with your ElevenLabs voice and model settings.

Dependencies:
    pip install requests sounddevice numpy
"""

from __future__ import annotations

import io
import logging
import os
import re
import wave

import numpy as np
import requests
import sounddevice as sd


ELEVENLABS_TTS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
LOGGER = logging.getLogger(__name__)


def _require_api_key() -> None:
    tts_api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not tts_api_key or "YOUR_" in tts_api_key:
        raise ValueError(
            "Add your ELEVENLABS_API_KEY in the .env file before running the project."
        )


def _get_tts_model() -> str:
    return os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_flash_v2_5")


def _get_voice_id() -> str:
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
    if not voice_id or "YOUR_" in voice_id:
        raise ValueError(
            "Add your ELEVENLABS_VOICE_ID in the .env file before running the project."
        )
    return voice_id


def _get_output_format() -> str:
    return os.getenv("ELEVENLABS_TTS_OUTPUT_FORMAT", "pcm_16000")


def _parse_pcm_sample_rate(output_format: str) -> int:
    match = re.fullmatch(r"pcm_(\d+)", output_format)
    if not match:
        raise ValueError(f"Unsupported PCM output format: {output_format}")
    return int(match.group(1))


def _play_pcm(audio_bytes: bytes, sample_rate: int) -> None:
    """Play raw PCM 16-bit mono audio returned by ElevenLabs."""
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    try:
        sd.play(audio_array, sample_rate)
        sd.wait()
    except Exception as exc:
        raise RuntimeError(
            "Audio playback failed. Check your output device and permissions."
        ) from exc


def _play_wav(audio_bytes: bytes) -> None:
    """Play WAV audio returned by ElevenLabs."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except Exception as exc:
        raise RuntimeError("Failed to decode WAV audio returned by ElevenLabs.") from exc

    if sample_width != 2:
        raise RuntimeError(
            "This demo expects 16-bit WAV audio from ElevenLabs."
        )

    audio_array = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio_array = audio_array.reshape(-1, channels)

    try:
        sd.play(audio_array, sample_rate)
        sd.wait()
    except Exception as exc:
        raise RuntimeError(
            "Audio playback failed. Check your output device and permissions."
        ) from exc


def speak_text(text: str) -> None:
    """
    Convert text to speech, then play the returned audio.

    This uses ElevenLabs TTS and plays PCM audio immediately.
    """
    _require_api_key()
    tts_api_key = os.getenv("ELEVENLABS_API_KEY", "")

    if not text.strip():
        raise ValueError("speak_text() received empty text.")

    headers = {
        "xi-api-key": tts_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": _get_tts_model(),
    }
    params = {
        "output_format": _get_output_format(),
    }
    LOGGER.info(
        "Sending TTS request to ElevenLabs with model_id=%s, voice_id=%s, output_format=%s.",
        _get_tts_model(),
        _get_voice_id(),
        params["output_format"],
    )

    try:
        response = requests.post(
            f"{ELEVENLABS_TTS_BASE_URL}/{_get_voice_id()}",
            headers=headers,
            json=payload,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"TTS request failed: {exc}") from exc

    output_format = _get_output_format()
    LOGGER.info("TTS response received from ElevenLabs (%s bytes).", len(response.content))

    if output_format.startswith("pcm_"):
        sample_rate = _parse_pcm_sample_rate(output_format)
        LOGGER.info("Playing raw PCM audio at %s Hz.", sample_rate)
        _play_pcm(response.content, sample_rate)
        return

    if output_format.startswith("wav_") or output_format == "wav":
        LOGGER.info("Playing WAV audio returned by ElevenLabs.")
        _play_wav(response.content)
        return

    raise RuntimeError(
        f"Unsupported ELEVENLABS_TTS_OUTPUT_FORMAT: {output_format}. "
        "Use a PCM format like pcm_24000 or a WAV format like wav_24000."
    )
