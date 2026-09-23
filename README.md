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
after `--pause` seconds of silence (default 2; raise it if it cuts you off mid-thought). One turn
can run up to `--max-turn` seconds of talking (default 300). Push-to-talk has no limit. It cannot interrupt the assistant yet; wait for
it to finish. Ctrl+C quits. In a `remember` session every exchange is appended to `transcripts/<stamp>.jsonl`; continue
one later with `--resume transcripts/<stamp>.jsonl`.

Options worth knowing:

- `--voice "..."` a description of the voice you want; it is designed once and cached in `voices/`
  (delete the `.wav` to re-roll). `--voice-file me.wav` clones a real recording instead (10-30 s,
  one speaker; it is transcribed automatically, or pass `--voice-text`).
- `--system "..."` the persona. The default asks for short spoken answers with no markdown.
- `--model`, `--ctx` (default 32k), `--think` (Qwen thinks first; slower), `--stt-model`.

## Memory (opt-in)

By default a session is off the record: nothing is read from memory, nothing is written, and no
transcript is kept. Start with `remember` to turn memory on for that session:

```bash
.venv/bin/python qwen_live.py remember
```

A `remember` session uses what it knows about you, can save new things, keeps a transcript in
`transcripts/`, and is summarized into memory when you quit. It doesn't paste its whole history into
every prompt. Everything lives in one local SQLite file, `memory.db` (git-ignored, never leaves the machine):

- **Memories**: short notes about you (facts, preferences, people, ongoing threads, events), each with
  an importance from 1 to 5.
- **Sessions**: one summary per conversation.
- **Exchanges**: every turn verbatim, searchable.

How it's used, cheapest first:

1. **Profile**: the ~10 most important memories and the last session's summary go in the system prompt
   once per session.
2. **Automatic recall**: each thing you say is keyword-searched against memory; clear matches (sharing
   two or more content words) are added for that one turn only. No extra model call.
3. **Tools**: Qwen can call `search_memory` (memories, session summaries, past exchanges) when you refer
   to something from before, and `remember` when you tell it something worth keeping ("remember that...").
   A search that finds nothing makes it say it doesn't remember rather than guess.
4. **Summaries**: when you quit (Ctrl+C) it summarizes the conversation and extracts new memories,
   reconciling them with existing ones (updated, not duplicated). Press Ctrl+C twice to skip; it's done
   on the next start. Long conversations fold their oldest turns into a running summary so the context
   never fills up.

Review and edit it yourself:

```bash
.venv/bin/python memory.py export          # everything, readable, in memory_export.md
.venv/bin/python memory.py list            # memories with their IDs
.venv/bin/python memory.py search "telescope"
.venv/bin/python memory.py forget 7        # remove one
.venv/bin/python memory.py edit 3 "new text"
.venv/bin/python memory.py add "Sam prefers short answers" --kind preference --importance 5
.venv/bin/python memory.py sessions        # then: memory.py show SESSION_ID
.venv/bin/python memory.py import          # summarize old transcripts/*.jsonl into memory
```

`--remember` is the same as `remember`; `--memory-db` points at another file.

## Bluetooth headsets

If the same Bluetooth headset is both your mic and your headphones, macOS switches it to its phone-call
profile whenever the mic opens: audio drops to mono and muffled, and anything still playing gets cut.
qwen_live waits for each reply to finish playing (plus `--tail` seconds, default 1) before it reopens
the mic, and prints a note at startup when it sees a shared device. For the best sound, use a separate
mic with `--input-device` and keep the headset for listening; if replies still get clipped, try
`--tail 1.5`.

## Latency

On an M1 Max with the 27B model: ~1 s to transcribe, ~1 s to first token, and Qwen3-TTS runs at
about 2x real time, so the first sentence starts roughly 3-4 s after you stop talking and the
rest keeps up. Shorter replies (the default system prompt asks for them) feel much more like a
conversation.
