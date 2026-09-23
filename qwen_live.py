#!/usr/bin/env python3
"""
qwen_live: talk to a local Qwen (Ollama) with a microphone and headphones, all on-device.

  mic -> Whisper (mlx-audio) -> Ollama /api/chat (streaming) -> Qwen3-TTS clone (mlx-audio) -> speakers

The reply is spoken sentence by sentence while the model is still writing, so the first words
arrive a few seconds after you stop talking. The assistant's voice is designed once from a text
description (or taken from a WAV you provide), saved in voices/, and cloned for every sentence so
it stays the same person.

Setup (Apple Silicon; same environment as the podcast project's .venv-tts):
  /opt/homebrew/bin/python3.12 -m venv .venv
  .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt

Run:
  .venv/bin/python qwen_live.py --check          # no microphone: TTS -> STT round trip + Ollama ping
  .venv/bin/python qwen_live.py                  # hands-free: speak, pause, it answers
  .venv/bin/python qwen_live.py --push-to-talk   # press Enter to start and stop each turn

Models are pulled from Hugging Face on first use (Whisper turbo ~1.6 GB, Qwen3-TTS ~3.5 GB each).
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

import memory as ltm

# mlx-audio prints progress on stdout; keep our own output readable
_real_stdout = sys.stdout

# --- Defaults (all overridable by flags) ---
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("QWEN_LIVE_MODEL", "qwen3.8:27b-mlx")
OLLAMA_NUM_CTX = int(os.environ.get("QWEN_LIVE_NUM_CTX", "32768"))
STT_MODEL = os.environ.get("QWEN_LIVE_STT_MODEL", "mlx-community/whisper-large-v3-turbo")
TTS_DESIGN_MODEL = os.environ.get("QWEN_LIVE_TTS_DESIGN_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
TTS_CLONE_MODEL = os.environ.get("QWEN_LIVE_TTS_CLONE_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
VOICES_DIR = Path("voices")
TRANSCRIPTS_DIR = Path("transcripts")

DEFAULT_VOICE_DESIGN = (
    "A warm, relaxed adult voice, American, mid-thirties; clear and natural, slightly wry, "
    "talking to a friend across a table, not performing."
)
REFERENCE_PASSAGE = (
    "Okay, so, I've been thinking about this since you mentioned it. It's not that the idea is wrong exactly, "
    "it's that it's incomplete. Let me try to say what I mean, and stop me if I'm going in circles."
)
DEFAULT_SYSTEM = (
    "You are a thoughtful, friendly conversational partner talking out loud with one person over headphones. "
    "Everything you write is spoken by a text-to-speech voice, so: plain spoken English, short sentences, no markdown, "
    "no lists, no headings, no emoji, no code unless asked. Keep replies brief, usually two to five sentences, and ask "
    "a question back when it helps. If you didn't catch something, say so."
)

STT_SR = 16000
TTS_SR = 24000


def log(msg: str):
    print(msg, file=_real_stdout, flush=True)


# --- Ollama ---

def ollama_check(url: str, model: str):
    try:
        r = httpx.get(f"{url}/api/tags", timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        sys.exit(f"Cannot reach Ollama at {url} ({exc}). Is it running?")
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if model not in names and f"{model}:latest" not in names:
        sys.exit(f"Ollama has no model '{model}'. Available: {', '.join(names) or 'none'}")


def ollama_stream(url: str, model: str, messages: list[dict], num_ctx: int, think: bool):
    """Yield content tokens as they arrive."""
    payload = {"model": model, "messages": messages, "stream": True, "think": think, "options": {"num_ctx": num_ctx}}
    with httpx.stream("POST", f"{url}/api/chat", json=payload, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break


def ollama_events(url: str, model: str, messages: list[dict], num_ctx: int, think: bool, tools: list | None = None):
    """Stream a reply: yields ("token", text) as it is written, and ("tools", [calls]) if the model calls tools."""
    payload = {"model": model, "messages": messages, "stream": True, "think": think, "options": {"num_ctx": num_ctx}}
    if tools:
        payload["tools"] = tools
    calls = []
    with httpx.stream("POST", f"{url}/api/chat", json=payload, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            msg = data.get("message", {})
            if msg.get("content"):
                yield "token", msg["content"]
            calls += msg.get("tool_calls") or []
            if data.get("done"):
                break
    if calls:
        yield "tools", calls


# --- Memory: what the model sees and the tools it can call ---

MEMORY_TOOLS = [
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search your long-term memory of this person and your past conversations with them. Use it when they "
                       "refer to something from before that isn't already in front of you ('remember when...', 'what did I say about...', "
                       "a name you don't recognise). Don't use it for general knowledge.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "a few keywords: names, places, topics"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Save something to long-term memory: when they ask you to remember something, or tell you a durable fact about "
                       "themselves, a person in their life, a preference, or an ongoing plan. One short third-person sentence.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "e.g. 'Sam's sister Maya is getting married in May.'"},
            "kind": {"type": "string", "enum": list(ltm.KINDS)},
            "importance": {"type": "integer", "description": "1-5; 5 if they explicitly asked you to remember it"}},
            "required": ["text"]}}},
]

MEMORY_SYSTEM = (
    "You have a long-term memory of this person. What you already know is listed below; more can be found with the "
    "search_memory tool, and remember saves something new. Use memory naturally, the way a friend would, without announcing "
    "it ('I remember you said...' is fine, 'according to my memory database' is not). Only search when they refer to "
    "something from before that isn't here. If a tool finds nothing, say you don't remember rather than guessing."
)


def memory_system_prompt(base: str, mem: "ltm.Memory") -> str:
    parts = [base, MEMORY_SYSTEM]
    profile = mem.profile()
    if profile:
        parts.append("What you know about them:\n" + profile)
    last = mem.last_session_summary()
    if last:
        parts.append("Last time you talked: " + last)
    return "\n\n".join(parts)


def drop_repeat(text: str, already: str) -> str:
    """If a post-tool round starts by repeating what was already said, drop the repeat."""
    norm = lambda t: re.sub(r"\W+", " ", clean_for_speech(t)).strip().lower()
    t, a = clean_for_speech(text), norm(already)
    if a and norm(t).startswith(a[:60]):
        cut = len(already)
        return t[cut:] if len(t) > cut else ""
    return t


def emit(value: str, reply: list, said: list, splitter, speaker):
    if not value:
        return
    reply.append(value)
    said.append(value)
    _real_stdout.write(value)
    _real_stdout.flush()
    for sentence in splitter.feed(value):
        speaker.say(sentence)


def run_tool(mem: "ltm.Memory", call: dict) -> tuple[str, str]:
    fn = call.get("function", {})
    name, args = fn.get("name", ""), fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"query": args, "text": args}
    if name == "search_memory":
        hits = mem.search(str(args.get("query", "")))
        log(f"  [memory search: {args.get('query')!r} -> {len(hits)} hit(s)]")
        if not hits:
            return name, "Nothing found in memory for that."
        return name, "\n".join(f"- ({h['type']}, {h['when']}) {h['text']}" for h in hits)
    if name == "remember":
        text = str(args.get("text", "")).strip()
        if not text:
            return name, "Nothing to remember."
        action, mem_id = mem.remember(text, str(args.get("kind", "note")), int(args.get("importance", 3) or 3))
        log(f"  [memory {action} #{mem_id}: {text}]")
        return name, f"Saved ({action})."
    return name, f"Unknown tool {name}."


# --- Sentence chunking for streaming speech ---

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


def clean_for_speech(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>?", "", text)   # stray fragments a model leaks after a tool call
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SentenceSplitter:
    """Feed tokens in, get whole sentences out as soon as they complete."""

    def __init__(self, min_chars: int = 24):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, token: str) -> list[str]:
        self.buf += token
        out = []
        while True:
            m = _SENTENCE_END.search(self.buf)
            if not m:
                break
            head, self.buf = self.buf[: m.end()], self.buf[m.end():]
            head = clean_for_speech(head)
            if head:
                out.append(head)
        # Merge fragments that are too short to be worth a TTS call
        merged = []
        for s in out:
            if merged and len(merged[-1]) < self.min_chars:
                merged[-1] = f"{merged[-1]} {s}"
            else:
                merged.append(s)
        return merged

    def flush(self) -> list[str]:
        rest, self.buf = clean_for_speech(self.buf), ""
        return [rest] if rest else []


# --- Speech models ---

class Speech:
    def __init__(self, stt_model: str, design_model: str, clone_model: str):
        self.stt_name, self.design_name, self.clone_name = stt_model, design_model, clone_model
        self._stt = self._design = self._clone = None
        self.mlx_lock = threading.Lock()   # MLX work from one thread at a time

    def stt(self):
        if self._stt is None:
            from mlx_audio.stt.utils import load_model
            log(f"loading STT {self.stt_name} ...")
            self._stt = load_model(self.stt_name)
            if "whisper" in self.stt_name.lower() and getattr(self._stt, "_processor", None) is None:
                # mlx-community Whisper conversions ship weights only; the tokenizer and feature
                # extractor come from OpenAI's original repo of the same name.
                from transformers import WhisperProcessor
                base = re.sub(r"-(q\d+|fp16|bf16|mlx|asr.*)$", "", self.stt_name.split("/")[-1])
                source = os.environ.get("QWEN_LIVE_WHISPER_PROCESSOR", f"openai/{base}")
                log(f"loading Whisper processor from {source} ...")
                self._stt._processor = WhisperProcessor.from_pretrained(source)
        return self._stt

    def design(self):
        if self._design is None:
            from mlx_audio.tts.utils import load_model
            log(f"loading voice design {self.design_name} ...")
            self._design = load_model(self.design_name)
        return self._design

    def clone(self):
        if self._clone is None:
            from mlx_audio.tts.utils import load_model
            log(f"loading TTS {self.clone_name} ...")
            self._clone = load_model(self.clone_name)
        return self._clone

    def transcribe(self, audio16k: np.ndarray) -> str:
        with self.mlx_lock:
            result = self.stt().generate(audio16k.astype(np.float32), language="en")
        return clean_for_speech(getattr(result, "text", str(result)))

    def _to_float(self, results) -> np.ndarray:
        chunks = [np.array(r.audio, dtype=np.float32).reshape(-1) for r in results]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        sr = results[0].sample_rate if results else TTS_SR
        if sr != TTS_SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, TTS_SR)
            audio = resample_poly(audio, TTS_SR // g, sr // g).astype(np.float32)
        return np.clip(audio, -1.0, 1.0)

    def design_voice(self, description: str, text: str) -> np.ndarray:
        with self.mlx_lock:
            results = list(self.design().generate_voice_design(text=text, instruct=description, language="english"))
        return self._to_float(results)

    def speak(self, text: str, ref_audio: np.ndarray, ref_text: str) -> np.ndarray:
        import mlx.core as mx
        with self.mlx_lock:
            results = list(self.clone().generate(text=text, ref_audio=mx.array(ref_audio), ref_text=ref_text, lang_code="english"))
        return self._to_float(results)


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit(f"{path}: need a 16-bit WAV")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def ensure_voice(speech: Speech, description: str, voice_file: str | None, voice_text: str | None) -> tuple[np.ndarray, str]:
    """Return (reference audio at 24 kHz, its transcript). A provided WAV wins; otherwise design once and cache."""
    if voice_file:
        audio, sr = read_wav(Path(voice_file))
        if sr != TTS_SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, TTS_SR)
            audio = resample_poly(audio, TTS_SR // g, sr // g).astype(np.float32)
        if not voice_text:
            log("transcribing the reference clip so the cloner knows what it says ...")
            from scipy.signal import resample_poly
            voice_text = speech.transcribe(resample_poly(audio, 2, 3).astype(np.float32))
        return audio, voice_text
    slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")[:40]
    import hashlib
    key = hashlib.sha256(description.encode()).hexdigest()[:10]
    wav = VOICES_DIR / f"{slug}-{key}.wav"
    if wav.exists():
        audio, _ = read_wav(wav)
        return audio, REFERENCE_PASSAGE
    log(f"designing the voice once ({description[:60]}...) ...")
    audio = speech.design_voice(description, REFERENCE_PASSAGE)
    write_wav(wav, audio, TTS_SR)
    wav.with_suffix(".txt").write_text(f"{description}\n\n{REFERENCE_PASSAGE}\n")
    log(f"saved {wav} (delete it to re-roll the voice)")
    return audio, REFERENCE_PASSAGE


# --- Microphone ---

class Mic:
    """Records one utterance: waits for speech, stops after a pause. Energy-based, no extra deps."""

    def __init__(self, device=None, start_threshold=0.015, stop_after=1.5, max_seconds=300.0, min_speech=0.25):
        self.device = device
        self.start_threshold = start_threshold
        self.stop_after = stop_after
        self.max_seconds = max_seconds
        self.min_speech = min_speech

    def calibrate(self, seconds: float = 0.6):
        import sounddevice as sd
        rec = sd.rec(int(seconds * STT_SR), samplerate=STT_SR, channels=1, dtype="float32", device=self.device)
        sd.wait()
        noise = float(np.sqrt(np.mean(rec[:, 0] ** 2)) + 1e-6)
        self.start_threshold = max(self.start_threshold, noise * 3.5)
        return noise

    def record(self) -> np.ndarray | None:
        import sounddevice as sd
        block = int(0.03 * STT_SR)
        q: queue.Queue = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        chunks, speaking, silence, spoken = [], False, 0.0, 0.0
        started = time.time()
        speech_started = None
        with sd.InputStream(samplerate=STT_SR, channels=1, dtype="float32", blocksize=block, device=self.device, callback=cb):
            while True:
                data = q.get()
                rms = float(np.sqrt(np.mean(data ** 2)))
                if not speaking:
                    if rms > self.start_threshold:
                        speaking = True
                        speech_started = time.time()
                        chunks.append(data)
                        spoken += len(data) / STT_SR
                    elif time.time() - started > 120:
                        return None   # two minutes of nothing; let the caller decide
                    continue
                chunks.append(data)
                if rms > self.start_threshold * 0.6:
                    silence = 0.0
                    spoken += len(data) / STT_SR
                else:
                    silence += len(data) / STT_SR
                # The cap counts from when you started talking, not from when it started listening
                if silence >= self.stop_after or (time.time() - speech_started) > self.max_seconds:
                    if silence < self.stop_after:
                        log(f"  (reached the {self.max_seconds:.0f}s limit for one turn; raise it with --max-turn)")
                    break
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        if spoken < self.min_speech:
            return None
        return audio


def push_to_talk_record(device=None) -> np.ndarray | None:
    import sounddevice as sd
    input("  [Enter] to start talking ... ")
    frames = []

    def cb(indata, n, t, status):
        frames.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=STT_SR, channels=1, dtype="float32", device=device, callback=cb):
        input("  recording, [Enter] to stop ")
    return np.concatenate(frames) if frames else None


# --- Playback pipeline: sentences -> TTS thread -> audio queue -> output stream thread ---

class Speaker:
    def __init__(self, speech: Speech, ref_audio: np.ndarray, ref_text: str, device=None, tail_seconds: float = 0.4):
        self.speech, self.ref_audio, self.ref_text, self.device = speech, ref_audio, ref_text, device
        self.tail_seconds = tail_seconds
        self.sentences: queue.Queue = queue.Queue()
        self.audio: queue.Queue = queue.Queue()
        self.first_audio_at: float | None = None
        self._turn_done = threading.Event()
        threading.Thread(target=self._tts_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def say(self, sentence: str):
        self.sentences.put(sentence)

    def end_turn(self):
        self.sentences.put(None)

    def _tts_loop(self):
        while True:
            s = self.sentences.get()
            if s is None:
                self.audio.put(None)
                continue
            try:
                self.audio.put(self.speech.speak(s, self.ref_audio, self.ref_text))
            except Exception as exc:
                log(f"  (tts failed on a sentence: {exc})")

    def _play_loop(self):
        import sounddevice as sd
        with sd.OutputStream(samplerate=TTS_SR, channels=1, dtype="float32", device=self.device) as out:
            while True:
                a = self.audio.get()
                if a is None:
                    # write() returns once audio is queued, not heard. Push a short tail of silence through,
                    # then wait out the device buffer, so the last words play before the mic reopens.
                    out.write(np.zeros((int(self.tail_seconds * TTS_SR), 1), np.float32))
                    time.sleep(float(out.latency) + self.tail_seconds)
                    self._turn_done.set()
                    continue
                if self.first_audio_at is None:
                    self.first_audio_at = time.time()
                out.write(a.reshape(-1, 1))
                out.write(np.zeros((int(0.12 * TTS_SR), 1), np.float32))   # small breath between sentences

    def begin_turn(self):
        self._turn_done = threading.Event()
        self.first_audio_at = None

    def wait(self):
        self._turn_done.wait()


def warn_shared_bluetooth(in_dev, out_dev):
    """A Bluetooth headset used as both mic and speaker drops to its low-quality call profile whenever
    the mic opens, which also cuts off whatever is still playing. Say so once."""
    import sounddevice as sd
    try:
        i = sd.query_devices(in_dev, kind="input")["name"]
        o = sd.query_devices(out_dev, kind="output")["name"]
    except Exception:
        return
    if i == o and not re.search(r"built-in|macbook|mac studio|imac|usb", i, re.I):
        log(f"note: '{i}' is both the mic and the headphones. If it's Bluetooth, opening its mic switches it to call quality\n"
            f"      (mono, muffled) and can clip the end of replies. Better: a separate mic (USB, webcam, or the Mac's own\n"
            f"      if it has one) via --input-device (see --list-devices). Otherwise raise --tail, e.g. --tail 1.0.")


# --- Main loop ---

def run_check(args, speech: Speech):
    """No microphone needed: design/load the voice, speak a line, transcribe it back, ping Ollama."""
    ollama_check(args.ollama_url, args.model)
    log("ollama: ok")
    ref_audio, ref_text = ensure_voice(speech, args.voice, args.voice_file, args.voice_text)
    line = "This is a quick check of the speech pipeline. If you can read this back, we are in business."
    t = time.time()
    audio = speech.speak(line, ref_audio, ref_text)
    log(f"tts: {len(audio) / TTS_SR:.1f}s of audio in {time.time() - t:.1f}s")
    write_wav(Path("check_tts.wav"), audio, TTS_SR)
    from scipy.signal import resample_poly
    t = time.time()
    heard = speech.transcribe(resample_poly(audio, 2, 3).astype(np.float32))
    log(f"stt ({time.time() - t:.1f}s): {heard}")
    t = time.time()
    reply = "".join(ollama_stream(args.ollama_url, args.model, [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": "Say hello in one short sentence."}], args.ctx, args.think))
    log(f"llm ({time.time() - t:.1f}s): {clean_for_speech(reply)}")
    log("check complete; saved check_tts.wav")


def main():
    ap = argparse.ArgumentParser(description="Talk to a local Qwen with a microphone and headphones. Nothing is saved unless you "
                                             "start it with 'remember'.")
    ap.add_argument("mode", nargs="?", choices=["remember"],
                    help="'remember': this session uses long-term memory and is saved (transcript, summary, memories). "
                         "Without it nothing is read from or written to memory, and no transcript is kept.")
    ap.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default {OLLAMA_MODEL})")
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--ctx", type=int, default=OLLAMA_NUM_CTX, help=f"context window (default {OLLAMA_NUM_CTX})")
    ap.add_argument("--think", action="store_true", help="let Qwen think before answering (slower)")
    ap.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt / persona")
    ap.add_argument("--voice", default=DEFAULT_VOICE_DESIGN, help="text description of the assistant's voice (designed once, cached in voices/)")
    ap.add_argument("--voice-file", help="WAV of a voice to clone instead of designing one (10-30 s, one speaker)")
    ap.add_argument("--voice-text", help="transcript of --voice-file (transcribed automatically if omitted)")
    ap.add_argument("--stt-model", default=STT_MODEL)
    ap.add_argument("--tts-design-model", default=TTS_DESIGN_MODEL)
    ap.add_argument("--tts-clone-model", default=TTS_CLONE_MODEL)
    ap.add_argument("--input-device", help="microphone name or index (see --list-devices)")
    ap.add_argument("--output-device", help="headphones/speaker name or index")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--push-to-talk", action="store_true", help="press Enter to start and stop each turn instead of auto-detecting pauses")
    ap.add_argument("--tail", type=float, default=0.4, help="extra seconds to let the last words finish before the mic reopens (default 0.4; raise it for Bluetooth headsets)")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds of silence that ends your turn (default 1.5; raise it if you get cut off mid-thought)")
    ap.add_argument("--max-turn", type=float, default=300.0, help="longest you can talk in one turn, in seconds (default 300)")
    ap.add_argument("--resume", help="transcript .jsonl to continue from")
    ap.add_argument("--check", action="store_true", help="no-microphone self test, then exit")
    ap.add_argument("--remember", action="store_true", help="same as the 'remember' mode")
    ap.add_argument("--memory-db", default=str(ltm.DB_PATH), help=f"memory database (default {ltm.DB_PATH}); manage it with memory.py")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    speech = Speech(args.stt_model, args.tts_design_model, args.tts_clone_model)
    if args.check:
        run_check(args, speech)
        return

    ollama_check(args.ollama_url, args.model)
    in_dev = int(args.input_device) if args.input_device and args.input_device.isdigit() else args.input_device
    out_dev = int(args.output_device) if args.output_device and args.output_device.isdigit() else args.output_device

    warn_shared_bluetooth(in_dev, out_dev)
    ref_audio, ref_text = ensure_voice(speech, args.voice, args.voice_file, args.voice_text)
    speech.stt()   # load now, not on the first utterance
    speaker = Speaker(speech, ref_audio, ref_text, device=out_dev, tail_seconds=args.tail)

    remembering = args.remember or args.mode == "remember"
    mem = ltm.Memory(Path(args.memory_db)) if remembering else None
    chat = ltm.ollama_chat_fn(args.ollama_url, args.model, args.ctx)
    session_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    if mem:
        mem.start_session(session_id)
        pending = [r["id"] for r in mem.sessions() if not r["summary"] and r["turns"] and r["id"] != session_id]
        for sid in pending:   # sessions that ended without a summary (crash, second Ctrl+C)
            log(f"summarizing unfinished session {sid} ...")
            msgs = [x for e in mem.session_exchanges(sid) for x in ({"role": "user", "content": e["user"]}, {"role": "assistant", "content": e["assistant"]})]
            try:
                ltm.summarize_session(mem, sid, msgs, chat, log)
            except Exception as exc:
                log(f"  (could not summarize {sid}: {exc})")
        profile_ids = mem.profile_ids()
        n_mem = mem.db.execute("SELECT count(*) FROM memories").fetchone()[0]
        log(f"memory: {n_mem} memories, {len(mem.sessions()) - 1} past sessions ({args.memory_db})")
    messages = [{"role": "system", "content": memory_system_prompt(args.system, mem) if mem else args.system}]
    if args.resume:
        for line in Path(args.resume).read_text().splitlines():
            if line.strip():
                messages.append(json.loads(line))
        log(f"resumed {len(messages) - 1} messages from {args.resume}")
    transcript = None
    if mem:
        TRANSCRIPTS_DIR.mkdir(exist_ok=True)
        transcript = TRANSCRIPTS_DIR / f"{session_id}.jsonl"

    mic = Mic(device=in_dev, stop_after=args.pause, max_seconds=args.max_turn)
    if not args.push_to_talk:
        noise = mic.calibrate()
        log(f"mic calibrated (noise floor {noise:.4f}, start threshold {mic.start_threshold:.4f})")
    if mem:
        log(f"\nTalking to {args.model}, REMEMBERING this session (transcript {transcript}). Ctrl+C to quit.\n")
    else:
        log(f"\nTalking to {args.model}, off the record: nothing is saved and no memory is used. "
            f"Start with 'remember' to change that. Ctrl+C to quit.\n")

    try:
        while True:
            log("listening ..." if not args.push_to_talk else "")
            audio = push_to_talk_record(in_dev) if args.push_to_talk else mic.record()
            if audio is None or len(audio) == 0:
                continue
            t0 = time.time()
            heard = speech.transcribe(audio)
            if not heard or len(heard) < 2:
                log("  (didn't catch that)")
                continue
            log(f"You ({time.time() - t0:.1f}s): {heard}")
            messages.append({"role": "user", "content": heard})

            if mem:
                before = len(messages)
                messages = ltm.compress_history(messages, chat)
                if len(messages) < before:
                    log("  [folded older turns into a summary to keep the context small]")

            # Automatic recall: memories that clearly match what was just said ride along for this turn only,
            # placed just before the new message so the cached prefix of the conversation stays valid.
            request = list(messages)
            if mem:
                hits = mem.recall(heard, exclude=profile_ids)
                if hits:
                    log(f"  [recalled {len(hits)} memory(ies)]")
                    note = "Possibly relevant from memory (use only if it fits):\n" + "\n".join(f"- {h['text']}" for h in hits)
                    request = request[:-1] + [{"role": "system", "content": note}, request[-1]]

            speaker.begin_turn()
            splitter = SentenceSplitter()
            reply, t1 = [], time.time()
            _real_stdout.write("Qwen: ")
            _real_stdout.flush()
            already = ""   # what has been spoken this turn, so a post-tool round can't repeat it
            for _round in range(3):   # a reply may call memory tools, then continue
                calls, said = [], []
                # After a tool result Qwen tends to draft its answer in "thinking" and leak it into the text;
                # with thinking on for those rounds the draft goes to its own field and the answer comes once.
                think = args.think or _round > 0
                pending = "" if _round > 0 and already else None   # hold the start until we know it isn't a repeat
                for kind, value in ollama_events(args.ollama_url, args.model, request, args.ctx, think,
                                                 tools=MEMORY_TOOLS if mem else None):
                    if kind == "tools":
                        calls = value
                        continue
                    if pending is not None:
                        pending += value
                        if len(pending) < len(already) + 10:
                            continue
                        value, pending = drop_repeat(pending, already), None
                    emit(value, reply, said, splitter, speaker)
                if pending:
                    emit(drop_repeat(pending, already), reply, said, splitter, speaker)
                already += "".join(said)
                if not calls:
                    break
                step = [{"role": "assistant", "content": "".join(said), "tool_calls": calls}]
                for call in calls:
                    name, result = run_tool(mem, call)
                    step.append({"role": "tool", "tool_name": name, "content": result})
                request += step
                messages += step
            for sentence in splitter.flush():
                speaker.say(sentence)
            speaker.end_turn()
            _real_stdout.write("\n")
            full = clean_for_speech("".join(reply))
            messages.append({"role": "assistant", "content": full})
            if transcript:
                with transcript.open("a") as f:
                    f.write(json.dumps({"role": "user", "content": heard}) + "\n" + json.dumps(messages[-1]) + "\n")
            if mem:
                mem.log_exchange(session_id, heard, full)
            speaker.wait()
            first = f"{speaker.first_audio_at - t1:.1f}s to first audio" if speaker.first_audio_at else "no audio"
            log(f"  ({time.time() - t1:.1f}s total, {first})\n")
    except KeyboardInterrupt:
        if mem and mem.session_exchanges(session_id):
            log("\nsaving memories from this conversation (Ctrl+C again to skip; it'll be done next start) ...")
            try:
                res = ltm.summarize_session(mem, session_id, messages, chat, log)
                log(f"  {res['added']} new, {res['updated']} updated. {res['summary']}")
            except KeyboardInterrupt:
                pass
            except Exception as exc:
                log(f"  (couldn't summarize now: {exc}; it'll be done next start)")
        log("bye")


if __name__ == "__main__":
    main()
