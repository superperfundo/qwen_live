# qwen_live

Talk to a local Qwen out loud: microphone in, headphones out, nothing leaves the machine.

```
mic -> Whisper (mlx-audio) -> Ollama /api/chat, streaming -> Qwen3-TTS clone (mlx-audio) -> headphones
```

The reply is spoken sentence by sentence while the model is still writing, so you hear the first
words a few seconds after you stop talking. The assistant's voice is designed once from a text
description (or cloned from a WAV you give it), saved in `voices/`, and reused for every sentence.

## Setup (Apple Silicon)

Ollama running with a model pulled (default `qwen3.8:27b-mlx`), then:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
```

No separate speech-to-text install: Whisper comes with `mlx-audio`. The model weights download
from Hugging Face on first use: Whisper large-v3-turbo (~1.6 GB), Qwen3-TTS VoiceDesign and Base
(~3.5 GB each). If you already have the podcast project's `.venv-tts`, that environment works too.

## Use

```bash
.venv/bin/python qwen_live.py --check          # no mic: speaks a line, transcribes it back, pings Ollama
.venv/bin/python qwen_live.py                  # hands-free: talk, pause a second, it answers
.venv/bin/python qwen_live.py --push-to-talk   # Enter to start, Enter to stop, each turn
.venv/bin/python qwen_live.py --list-devices   # then --input-device / --output-device by name or number
```

Hands-free mode calibrates to the room's noise for half a second at startup and ends your turn
after `--pause` seconds of silence (default 1.0). It cannot interrupt the assistant yet; wait for
it to finish. Ctrl+C quits. Every exchange is appended to `transcripts/<stamp>.jsonl`; continue
one later with `--resume transcripts/<stamp>.jsonl`.

Options worth knowing:

- `--voice "..."` a description of the voice you want; it is designed once and cached in `voices/`
  (delete the `.wav` to re-roll). `--voice-file me.wav` clones a real recording instead (10-30 s,
  one speaker; it is transcribed automatically, or pass `--voice-text`).
- `--system "..."` the persona. The default asks for short spoken answers with no markdown.
- `--model`, `--ctx` (default 32k), `--think` (Qwen thinks first; slower), `--stt-model`.

## Bluetooth headsets

If the same Bluetooth headset is both your mic and your headphones, macOS switches it to its phone-call
profile whenever the mic opens: audio drops to mono and muffled, and anything still playing gets cut.
qwen_live waits for each reply to finish playing (plus `--tail` seconds, default 0.4) before it reopens
the mic, and prints a note at startup when it sees a shared device. For the best sound, use a separate
mic with `--input-device` and keep the headset for listening; if the headset is the only mic, try
`--tail 1.0`.

## Latency

On an M1 Max with the 27B model: ~1 s to transcribe, ~1 s to first token, and Qwen3-TTS runs at
about 2x real time, so the first sentence starts roughly 3-4 s after you stop talking and the
rest keeps up. Shorter replies (the default system prompt asks for them) feel much more like a
conversation.
