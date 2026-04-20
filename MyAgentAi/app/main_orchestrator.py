"""
Minimal voice agent demo.

This file connects the full pipeline:
1. Record microphone audio
2. Convert speech to text
3. Ask the LLM for a reply
4. Convert the reply to speech
5. Play the reply back

Quick start:
    pip install requests sounddevice numpy
    python main.py

Before running:
    - Add your ElevenLabs and Gemini settings in the .env file

Notes:
    - This is a learning project, not a production system.
    - STT and TTS use ElevenLabs by default, while the LLM uses Gemini.
    - You can still swap providers later without changing main.py.
"""

from __future__ import annotations

import logging
import os
import re
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


NON_SPEECH_PATTERNS = {
    "[pause]",
    "[silence]",
    "[noise]",
    "[music]",
    "[laughter]",
    "[clears throat]",
    "[cough]",
    "[breathing]",
}


def load_env_file(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        logging.warning("No .env file found at %s", env_path)
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def run_once() -> None:
    from llm import generate_response
    from stt import record_audio, transcribe_audio
    from tts import speak_text

    total_start = time.perf_counter()

    logging.info("Recording from microphone...")
    recording_start = time.perf_counter()
    audio_bytes = record_audio()
    logging.info("Recording step completed in %.2fs.", time.perf_counter() - recording_start)

    logging.info("Transcribing audio...")
    stt_start = time.perf_counter()
    user_text = transcribe_audio(audio_bytes).strip()
    logging.info("STT step completed in %.2fs.", time.perf_counter() - stt_start)
    if not user_text:
        print("I did not hear anything clear enough to transcribe. Please try again.")
        return

    normalized_text = re.sub(r"\s+", " ", user_text.strip().lower())
    if normalized_text in NON_SPEECH_PATTERNS:
        logging.info("Ignoring non-speech transcript returned by STT: %s", user_text)
        print(f"\nYou: {user_text}")
        print("Assistant: Je n'ai entendu qu'un bruit ou une pause. Parle-moi normalement et je te repondrai.\n")
        return

    print(f"\nYou: {user_text}")

    logging.info("Generating LLM response...")
    llm_start = time.perf_counter()
    reply_text = generate_response(user_text).strip()
    logging.info("LLM step completed in %.2fs.", time.perf_counter() - llm_start)
    if not reply_text:
        print("The LLM returned an empty response.")
        return

    print(f"Assistant: {reply_text}\n")

    logging.info("Playing spoken response...")
    tts_start = time.perf_counter()
    speak_text(reply_text)
    logging.info("TTS step completed in %.2fs.", time.perf_counter() - tts_start)
    logging.info("Full pipeline completed in %.2fs.", time.perf_counter() - total_start)


def main() -> None:
    """Simple CLI loop for the voice assistant."""
    load_env_file()

    print("Minimal Voice Agent")
    print("Press Enter to talk, or type 'q' and press Enter to quit.\n")

    while True:
        try:
            user_input = input("> ")
            if user_input.strip().lower() in {"q", "quit", "exit"}:
                print("Goodbye!")
                break

            run_once()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            logging.exception("The voice pipeline failed.")
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
