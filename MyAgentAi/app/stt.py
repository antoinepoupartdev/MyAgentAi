"""
Speech-to-text utilities.

This module has two jobs:
1. Record a short WAV clip from the microphone
2. Send that WAV clip to an STT API

Configuration:
    Update the .env file with your ElevenLabs API details.

Dependencies:
    pip install requests sounddevice numpy
"""

from __future__ import annotations

import io
import logging
import os
import time
import wave

import numpy as np
import requests
import sounddevice as sd


# Simple recording defaults for a voice demo.
DEFAULT_SECONDS = 5
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_SECONDS = 0.25
VOICE_THRESHOLD = 500
INITIAL_SILENCE_TIMEOUT_SECONDS = 3.0
END_SILENCE_SECONDS = 1.0
MIN_SPEECH_SECONDS = 0.5

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
LOGGER = logging.getLogger(__name__)


def _require_api_key() -> None:
    stt_api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not stt_api_key or "YOUR_" in stt_api_key:
        raise ValueError(
            "Add your ELEVENLABS_API_KEY in the .env file before running the project."
        )


def _get_stt_model() -> str:
    return os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v2")


def _get_max_record_seconds() -> float:
    return float(os.getenv("STT_MAX_RECORD_SECONDS", str(DEFAULT_SECONDS)))


def _get_chunk_duration_seconds() -> float:
    return float(os.getenv("STT_CHUNK_DURATION_SECONDS", str(CHUNK_DURATION_SECONDS)))


def _get_voice_threshold() -> int:
    return int(os.getenv("STT_VOICE_THRESHOLD", str(VOICE_THRESHOLD)))


def _get_initial_silence_timeout_seconds() -> float:
    return float(
        os.getenv(
            "STT_INITIAL_SILENCE_TIMEOUT_SECONDS",
            str(INITIAL_SILENCE_TIMEOUT_SECONDS),
        )
    )


def _get_end_silence_seconds() -> float:
    return float(os.getenv("STT_END_SILENCE_SECONDS", str(END_SILENCE_SECONDS)))


def _get_min_speech_seconds() -> float:
    return float(os.getenv("STT_MIN_SPEECH_SECONDS", str(MIN_SPEECH_SECONDS)))


def _get_preferred_input_device() -> int | None:
    """
    Return a working input device index.

    If STT_INPUT_DEVICE is set in the .env file, it is used first.
    Otherwise, choose the first device that exposes input channels.
    """
    configured_device = os.getenv("STT_INPUT_DEVICE", "").strip()
    if configured_device:
        try:
            device_index = int(configured_device)
            device_info = sd.query_devices(device_index)
            max_input_channels = int(device_info["max_input_channels"])
            if max_input_channels > 0:
                LOGGER.info(
                    "Using configured input device %s: %s",
                    device_index,
                    device_info["name"],
                )
                return device_index

            LOGGER.warning(
                "Configured STT_INPUT_DEVICE=%s is not an input device. Falling back to auto selection.",
                device_index,
            )
        except Exception as exc:
            LOGGER.warning(
                "Configured STT_INPUT_DEVICE=%s could not be opened: %s. Falling back to auto selection.",
                configured_device,
                exc,
            )

    try:
        default_input, _ = sd.default.device
        if isinstance(default_input, int) and default_input >= 0:
            device_info = sd.query_devices(default_input)
            max_input_channels = int(device_info["max_input_channels"])
            if max_input_channels > 0:
                LOGGER.info(
                    "Using default input device %s: %s",
                    default_input,
                    device_info["name"],
                )
                return default_input
    except Exception as exc:
        LOGGER.warning("Default input device is unavailable: %s", exc)

    devices = sd.query_devices()
    for index, device_info in enumerate(devices):
        if int(device_info["max_input_channels"]) > 0:
            LOGGER.info(
                "Falling back to detected input device %s: %s",
                index,
                device_info["name"],
            )
            return index

    raise RuntimeError(
        "No input microphone was found. Connect a microphone or set STT_INPUT_DEVICE in the .env file."
    )


def record_audio(
    seconds: int = DEFAULT_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> bytes:
    """
    Record microphone audio and return it as WAV bytes.

    Returning bytes keeps main.py decoupled from audio file handling.
    """
    max_record_seconds = _get_max_record_seconds()
    chunk_duration_seconds = _get_chunk_duration_seconds()
    voice_threshold = _get_voice_threshold()
    initial_silence_timeout_seconds = _get_initial_silence_timeout_seconds()
    end_silence_seconds = _get_end_silence_seconds()
    min_speech_seconds = _get_min_speech_seconds()

    total_frames = int(max_record_seconds * sample_rate)
    blocksize = max(1, int(sample_rate * chunk_duration_seconds))
    collected_chunks = []
    voice_detected = False
    speech_active = False
    device_index = _get_preferred_input_device()
    silent_chunks_after_speech = 0
    speech_chunks = 0
    required_silent_chunks = max(1, int(end_silence_seconds / chunk_duration_seconds))
    minimum_speech_chunks = max(1, int(min_speech_seconds / chunk_duration_seconds))
    max_initial_silence_chunks = max(
        1, int(initial_silence_timeout_seconds / chunk_duration_seconds)
    )

    print(
        f"Recording... Speak now. "
        f"(max {max_record_seconds:.1f}s, auto-stop after {end_silence_seconds:.1f}s of silence)"
    )
    LOGGER.info("Opening microphone stream on input device %s.", device_index)

    try:
        with sd.InputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
        ) as stream:
            LOGGER.info("Microphone stream opened.")
            frames_remaining = total_frames
            chunk_index = 0

            while frames_remaining > 0:
                frames_to_read = min(blocksize, frames_remaining)
                chunk, overflowed = stream.read(frames_to_read)
                if overflowed:
                    LOGGER.warning("Audio input overflow detected while recording.")

                collected_chunks.append(chunk.copy())
                frames_remaining -= frames_to_read
                chunk_index += 1

                peak_level = int(np.max(np.abs(chunk)))
                current_time = round(chunk_index * chunk_duration_seconds, 2)
                is_speaking = peak_level >= voice_threshold

                if is_speaking and not speech_active:
                    speech_active = True
                    voice_detected = True
                    silent_chunks_after_speech = 0
                    LOGGER.info(
                        "Voice detected at %.2fs (peak=%s, threshold=%s).",
                        current_time,
                        peak_level,
                        voice_threshold,
                    )
                elif is_speaking and speech_active:
                    silent_chunks_after_speech = 0
                    speech_chunks += 1
                elif not is_speaking and speech_active:
                    speech_active = False
                    LOGGER.info(
                        "Voice ended at %.2fs (peak=%s).",
                        current_time,
                        peak_level,
                    )
                    silent_chunks_after_speech = 1
                elif not is_speaking and voice_detected:
                    silent_chunks_after_speech += 1

                if is_speaking:
                    speech_chunks += 1

                if not voice_detected and chunk_index >= max_initial_silence_chunks:
                    LOGGER.info(
                        "No speech detected after %.2fs. Stopping early.",
                        current_time,
                    )
                    break

                if (
                    voice_detected
                    and speech_chunks >= minimum_speech_chunks
                    and silent_chunks_after_speech >= required_silent_chunks
                ):
                    LOGGER.info(
                        "Detected %.2fs of trailing silence after speech. Stopping recording early.",
                        silent_chunks_after_speech * chunk_duration_seconds,
                    )
                    break

                time.sleep(0.01)
    except Exception as exc:
        raise RuntimeError(
            "Microphone recording failed. Check your audio device and permissions."
        ) from exc

    if speech_active:
        LOGGER.info("Voice was still active when recording ended.")

    audio = np.concatenate(collected_chunks, axis=0)

    if not np.any(audio):
        raise RuntimeError("No microphone input was detected during recording.")

    if not voice_detected:
        LOGGER.warning(
            "No voice peak crossed the threshold during recording. "
            "The microphone captured audio, but speech may have been too quiet."
        )
    else:
        LOGGER.info("Recording finished with detectable speech.")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # int16 = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return buffer.getvalue()


def transcribe_audio(audio_bytes: bytes) -> str:
    _require_api_key()
    stt_api_key = os.getenv("ELEVENLABS_API_KEY", "")

    headers = {"xi-api-key": stt_api_key}
    data = {"model_id": _get_stt_model()}
    files = {"file": ("microphone.wav", audio_bytes, "audio/wav")}

    try:
        response = requests.post(
            ELEVENLABS_STT_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"STT request failed: {exc}") from exc

    payload = response.json()
    LOGGER.info("STT response received from ElevenLabs.")
    transcript = payload.get("text", "")
    if not transcript:
        raise RuntimeError(
            "STT succeeded but no transcript text was returned. "
            "If you changed providers, update transcribe_audio() to match "
            "the provider's response format."
        )

    return transcript
