#!/usr/bin/env python3
"""
Long-term memory for qwen_live: a small SQLite database with full-text search.

What is stored (memory.db):
  memories   short durable notes: facts about you, preferences, people, ongoing threads, events,
             and anything you asked it to remember. Each has a kind and an importance (1-5).
  sessions   one row per conversation, with a short summary written when it ends.
  exchanges  every turn (what you said, what it said), searchable, so exact wording can be found.

How qwen_live uses it, cheapest first:
  profile()        the ~10 most important memories, put in the system prompt once per session
  recall(text)     a keyword search on what you just said; strong matches ride along for one turn
  search(query)    the search_memory tool the model can call when you refer to the past
  remember(...)    the remember tool, for "remember that..." and things worth keeping
  summarize_session(...)  at the end of a session: a summary plus new memories, reconciled with
                          existing ones so they are updated rather than duplicated

CLI:
  memory.py list [--kind K]          memory.py search "query"
  memory.py add "text" [--kind K] [--importance N]
  memory.py forget ID                memory.py edit ID "new text"
  memory.py sessions                 memory.py show SESSION_ID
  memory.py import [transcripts/*.jsonl]   summarize past transcripts not yet in memory
  memory.py stats
  memory.py export [FILE]            everything, human-readable, in one Markdown file for review
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("memory.db")
KINDS = ("fact", "preference", "person", "thread", "event", "note")
PROFILE_SIZE = 10
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just let me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves yeah okay ok um uh like really just gonna wanna kind sort thing
things stuff get got go going know think mean say said tell told want well oh hey hi hello right
""".split())


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def terms(text: str) -> list[str]:
    """Content words of a text, lowercased, stopwords removed, in order, unique."""
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        w = w.strip("'")
        if len(w) > 2 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: content words, prefix-matched, ORed."""
    return " OR ".join(f'"{t}"*' for t in terms(text)[:12])


def jaccard(a: str, b: str) -> float:
    sa, sb = set(terms(a)), set(terms(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


class Memory:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY, created TEXT, updated TEXT, kind TEXT, text TEXT,
                importance INTEGER DEFAULT 3, source TEXT, uses INTEGER DEFAULT 0, last_used TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, started TEXT, ended TEXT, turns INTEGER DEFAULT 0, summary TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(summary, content='sessions', content_rowid='rowid');
            CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
                INSERT INTO sessions_fts(rowid, summary) VALUES (new.rowid, coalesce(new.summary, '')); END;
            CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE OF summary ON sessions BEGIN
                INSERT INTO sessions_fts(sessions_fts, rowid, summary) VALUES ('delete', old.rowid, coalesce(old.summary, ''));
                INSERT INTO sessions_fts(rowid, summary) VALUES (new.rowid, coalesce(new.summary, '')); END;

            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY, session TEXT, ts TEXT, user TEXT, assistant TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS exchanges_fts USING fts5(user, assistant, content='exchanges', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS exchanges_ai AFTER INSERT ON exchanges BEGIN
                INSERT INTO exchanges_fts(rowid, user, assistant) VALUES (new.id, new.user, new.assistant); END;
        """)
        self.db.commit()

    # --- writing ---

    def add(self, text: str, kind: str = "note", importance: int = 3, source: str = "user") -> int:
        kind = kind if kind in KINDS else "note"
        importance = max(1, min(5, int(importance)))
        cur = self.db.execute(
            "INSERT INTO memories (created, updated, kind, text, importance, source) VALUES (?, ?, ?, ?, ?, ?)",
            (now(), now(), kind, text.strip(), importance, source))
        self.db.commit()
        return cur.lastrowid

    def update(self, mem_id: int, text: str | None = None, importance: int | None = None, kind: str | None = None):
        row = self.get(mem_id)
        if not row:
            raise KeyError(mem_id)
        self.db.execute("UPDATE memories SET text = ?, importance = ?, kind = ?, updated = ? WHERE id = ?",
                        (text or row["text"], importance or row["importance"], kind or row["kind"], now(), mem_id))
        self.db.commit()

    def forget(self, mem_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self.db.commit()
        return cur.rowcount > 0

    def remember(self, text: str, kind: str = "note", importance: int = 3, source: str = "conversation") -> tuple[str, int]:
        """Add a memory unless a near-duplicate exists, in which case refresh that one. Returns (action, id)."""
        for row in self.similar(text, limit=3):
            if jaccard(row["text"], text) >= 0.6:
                self.update(row["id"], text=text if len(text) >= len(row["text"]) else None,
                            importance=max(row["importance"], importance))
                return "updated", row["id"]
        return "added", self.add(text, kind, importance, source)

    def start_session(self, session_id: str):
        self.db.execute("INSERT OR IGNORE INTO sessions (id, started) VALUES (?, ?)", (session_id, now()))
        self.db.commit()

    def log_exchange(self, session_id: str, user: str, assistant: str):
        self.db.execute("INSERT INTO exchanges (session, ts, user, assistant) VALUES (?, ?, ?, ?)", (session_id, now(), user, assistant))
        self.db.execute("UPDATE sessions SET turns = turns + 1, ended = ? WHERE id = ?", (now(), session_id))
        self.db.commit()

    def set_summary(self, session_id: str, summary: str):
        self.db.execute("UPDATE sessions SET summary = ?, ended = coalesce(ended, ?) WHERE id = ?", (summary, now(), session_id))
        self.db.commit()

    # --- reading ---

    def get(self, mem_id: int):
        return self.db.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()

    def list(self, kind: str | None = None) -> list:
        q, args = "SELECT * FROM memories", ()
        if kind:
            q, args = q + " WHERE kind = ?", (kind,)
        return self.db.execute(q + " ORDER BY importance DESC, updated DESC", args).fetchall()

    def similar(self, text: str, limit: int = 5) -> list:
        q = fts_query(text)
        if not q:
            return []
        return self.db.execute(
            "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?",
            (q, limit)).fetchall()

    def profile(self, limit: int = PROFILE_SIZE) -> str:
        """The most important standing memories, for the system prompt. Stable within a session."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE kind IN ('fact','preference','person','thread') "
            "ORDER BY importance DESC, uses DESC, updated DESC LIMIT ?", (limit,)).fetchall()
        return "\n".join(f"- {r['text']}" for r in rows)

    def last_session_summary(self) -> str:
        row = self.db.execute("SELECT * FROM sessions WHERE summary IS NOT NULL AND summary != '' ORDER BY ended DESC LIMIT 1").fetchone()
        return f"({row['ended']}) {row['summary']}" if row else ""

    def search(self, query: str, limit: int = 6) -> list[dict]:
        """Memories, session summaries and past exchanges, best first. Memories are weighted up."""
        q = fts_query(query)
        if not q:
            return []
        hits = []
        for r in self.db.execute(
                "SELECT m.id, m.text, m.kind, m.updated, bm25(memories_fts) AS s FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
                "WHERE memories_fts MATCH ? ORDER BY s LIMIT ?", (q, limit)):
            hits.append({"type": "memory", "id": r["id"], "when": r["updated"], "text": r["text"], "score": r["s"] * 1.5})
        for r in self.db.execute(
                "SELECT s.id, s.ended, s.summary, bm25(sessions_fts) AS sc FROM sessions_fts JOIN sessions s ON s.rowid = sessions_fts.rowid "
                "WHERE sessions_fts MATCH ? ORDER BY sc LIMIT ?", (q, limit)):
            hits.append({"type": "session", "id": r["id"], "when": r["ended"], "text": r["summary"], "score": r["sc"]})
        for r in self.db.execute(
                "SELECT e.id, e.ts, e.user, e.assistant, bm25(exchanges_fts) AS sc FROM exchanges_fts JOIN exchanges e ON e.id = exchanges_fts.rowid "
                "WHERE exchanges_fts MATCH ? ORDER BY sc LIMIT ?", (q, limit)):
            hits.append({"type": "exchange", "id": r["id"], "when": r["ts"],
                         "text": f"You said: {r['user'][:220]} | I said: {r['assistant'][:220]}", "score": r["sc"]})
        hits.sort(key=lambda h: h["score"])      # bm25: lower is better
        for h in hits:
            if h["type"] == "memory":
                self.db.execute("UPDATE memories SET uses = uses + 1, last_used = ? WHERE id = ?", (now(), h["id"]))
        self.db.commit()
        return hits[:limit]

    def recall(self, text: str, exclude: set[int] | None = None, limit: int = 3) -> list[dict]:
        """Cheap automatic recall on what was just said: only memories that share at least two content
        words with it (or one, if it's a short utterance), skipping ones already in the profile."""
        want = terms(text)
        if not want:
            return []
        need = 1 if len(want) <= 3 else 2
        out = []
        for r in self.similar(text, limit=8):
            if exclude and r["id"] in exclude:
                continue
            overlap = len(set(want) & set(terms(r["text"])))
            if overlap >= need:
                out.append({"id": r["id"], "text": r["text"], "when": r["updated"]})
            if len(out) >= limit:
                break
        return out

    def profile_ids(self, limit: int = PROFILE_SIZE) -> set[int]:
        return {r["id"] for r in self.db.execute(
            "SELECT id FROM memories WHERE kind IN ('fact','preference','person','thread') "
            "ORDER BY importance DESC, uses DESC, updated DESC LIMIT ?", (limit,))}

    def sessions(self) -> list:
        return self.db.execute("SELECT * FROM sessions ORDER BY started DESC").fetchall()

    def session_exchanges(self, session_id: str) -> list:
        return self.db.execute("SELECT * FROM exchanges WHERE session = ? ORDER BY id", (session_id,)).fetchall()


# --- summarization (needs a chat function: messages -> text) ---

EXTRACT_PROMPT = """Below is a spoken conversation between a person (USER) and an AI assistant (ASSISTANT).
Write:
1. "summary": 2-4 plain sentences on what was talked about and anything left open, written so it can be searched later.
2. "memories": durable things worth remembering about the USER for future conversations: facts about them, their
   preferences, people in their life (by name), ongoing projects or threads, notable events, and anything they asked
   to be remembered. Each is one short third-person sentence ("Sam has a dog named Biscuit."). Skip small talk,
   anything about the assistant itself, and anything only true for this conversation. Importance 1-5 (5 = core identity
   or explicitly asked to remember). Zero memories is fine.
Return JSON only: {"summary": "...", "memories": [{"text": "...", "kind": one of fact|preference|person|thread|event|note, "importance": 1-5}]}

CONVERSATION:
"""

RECONCILE_PROMPT = """You maintain a memory store about a person. For each NEW memory, decide against the EXISTING ones it resembles:
"add" (genuinely new), "update" (replaces or refines existing id N; give the merged text), or "skip" (already known).
Return JSON only: {"actions": [{"new": index, "action": "add"|"update"|"skip", "id": N or null, "text": merged text or null}]}

EXISTING:
{existing}

NEW:
{new}
"""


def parse_json(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    return json.loads(raw)


def transcript_text(messages: list[dict], limit_chars: int = 24000) -> str:
    lines = []
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            lines.append(f"{m['role'].upper()}: {m['content']}")
    text = "\n".join(lines)
    return text[-limit_chars:]


def summarize_session(mem: Memory, session_id: str, messages: list[dict], chat, log=print) -> dict:
    """Summarize a finished conversation into the session row and reconciled memories."""
    convo = transcript_text(messages)
    if len(convo) < 40:
        return {"summary": "", "added": 0, "updated": 0}
    data = parse_json(chat([{"role": "user", "content": EXTRACT_PROMPT + convo}], json_mode=True))
    summary = str(data.get("summary", "")).strip()
    mem.start_session(session_id)
    mem.set_summary(session_id, summary)
    new = [m for m in data.get("memories", []) if isinstance(m, dict) and str(m.get("text", "")).strip()]
    added = updated = 0
    if new:
        related = {}
        for m in new:
            for r in mem.similar(m["text"], limit=3):
                related[r["id"]] = r
        actions = None
        if related:
            existing = "\n".join(f"id {r['id']}: {r['text']}" for r in related.values())
            listing = "\n".join(f"{i}: {m['text']}" for i, m in enumerate(new))
            try:
                actions = parse_json(chat([{"role": "user", "content": RECONCILE_PROMPT.replace("{existing}", existing).replace("{new}", listing)}],
                                          json_mode=True)).get("actions")
            except Exception as exc:
                log(f"  (memory reconcile failed, falling back to similarity: {exc})")
        by_index = {a.get("new"): a for a in (actions or []) if isinstance(a, dict)}
        for i, m in enumerate(new):
            kind, imp = str(m.get("kind", "note")), int(m.get("importance", 3) or 3)
            act = by_index.get(i)
            if act and act.get("action") == "skip":
                continue
            if act and act.get("action") == "update" and act.get("id") in related:
                mem.update(act["id"], text=str(act.get("text") or m["text"]), importance=max(imp, related[act["id"]]["importance"]))
                updated += 1
                continue
            what, _ = mem.remember(str(m["text"]), kind, imp, source=f"session {session_id}")
            added += what == "added"
            updated += what == "updated"
    return {"summary": summary, "added": added, "updated": updated}


def compress_history(messages: list[dict], chat, budget_chars: int = 40000, keep_recent: int = 8) -> list[dict]:
    """Keep a long conversation inside the context window: fold the oldest turns into a running
    summary message, keeping the system prompt and the most recent turns verbatim."""
    body = [m for m in messages[1:] if m.get("role") in ("user", "assistant", "tool") or m.get("summary")]
    if sum(len(m.get("content") or "") for m in body) <= budget_chars or len(body) <= keep_recent + 2:
        return messages
    old, recent = body[:-keep_recent], body[-keep_recent:]
    prior = next((m["content"] for m in old if m.get("summary")), "")
    text = transcript_text([m for m in old if not m.get("summary")])
    summary = chat([{"role": "user", "content":
                     "Summarize this earlier part of a spoken conversation in 5-8 plain sentences, keeping names, facts, "
                     "decisions and open questions; no preamble.\n\n" + (f"EARLIER SUMMARY: {prior}\n\n" if prior else "") + text}],
                   json_mode=False).strip()
    note = {"role": "system", "summary": True, "content": f"Earlier in this conversation (summarized): {summary}"}
    return [messages[0], note] + recent


# --- CLI ---

def import_transcripts(mem: Memory, paths: list[Path], chat):
    done = {r["id"] for r in mem.sessions() if r["summary"]}
    for path in paths:
        sid = path.stem
        if sid in done:
            print(f"{sid}: already in memory")
            continue
        messages = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        mem.start_session(sid)
        pairs = [(messages[i]["content"], messages[i + 1]["content"]) for i in range(0, len(messages) - 1, 2)
                 if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant"]
        if not mem.session_exchanges(sid):
            for u, a in pairs:
                mem.log_exchange(sid, u, a)
        res = summarize_session(mem, sid, messages, chat)
        print(f"{sid}: {len(pairs)} exchanges, +{res['added']} memories, {res['updated']} updated. {res['summary'][:120]}")


def ollama_chat_fn(url: str, model: str, num_ctx: int = 32768):
    """Non-streaming chat for summaries. Use the same num_ctx as the conversation, or Ollama reloads the model."""
    import httpx

    def chat(messages, json_mode=False):
        payload = {"model": model, "messages": messages, "stream": False, "think": False, "options": {"num_ctx": num_ctx}}
        if json_mode:
            payload["format"] = "json"
        r = httpx.post(f"{url}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]
    return chat


def main():
    import os
    ap = argparse.ArgumentParser(description="Inspect and edit qwen_live's long-term memory.")
    ap.add_argument("--db", default=str(DB_PATH))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--kind", choices=KINDS)
    p = sub.add_parser("search"); p.add_argument("query")
    p = sub.add_parser("add"); p.add_argument("text"); p.add_argument("--kind", default="note", choices=KINDS); p.add_argument("--importance", type=int, default=4)
    p = sub.add_parser("forget"); p.add_argument("id", type=int)
    p = sub.add_parser("edit"); p.add_argument("id", type=int); p.add_argument("text")
    sub.add_parser("sessions")
    p = sub.add_parser("show"); p.add_argument("session")
    p = sub.add_parser("import"); p.add_argument("paths", nargs="*")
    p.add_argument("--model", default=os.environ.get("QWEN_LIVE_MODEL", "qwen3.8:27b-mlx"))
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    sub.add_parser("stats")
    p = sub.add_parser("export"); p.add_argument("file", nargs="?", default="memory_export.md")
    args = ap.parse_args()
    mem = Memory(Path(args.db))

    if args.cmd == "list":
        for r in mem.list(args.kind):
            print(f"{r['id']:>4}  [{r['kind']:<10} {r['importance']}]  {r['text']}   ({r['updated']}, used {r['uses']}x)")
    elif args.cmd == "search":
        for h in mem.search(args.query):
            print(f"{h['type']:<8} {h['when'] or '':<16} {h['text'][:200]}")
    elif args.cmd == "add":
        print(f"added memory {mem.add(args.text, args.kind, args.importance, 'cli')}")
    elif args.cmd == "forget":
        print("forgotten" if mem.forget(args.id) else f"no memory {args.id}")
    elif args.cmd == "edit":
        mem.update(args.id, text=args.text)
        print("updated")
    elif args.cmd == "sessions":
        for r in mem.sessions():
            print(f"{r['id']}  {r['turns']:>3} turns  {r['started']}  {(r['summary'] or '(not summarized)')[:110]}")
    elif args.cmd == "show":
        for e in mem.session_exchanges(args.session):
            print(f"[{e['ts']}]\nYOU: {e['user']}\nQWEN: {e['assistant']}\n")
    elif args.cmd == "import":
        url = args.ollama_url if args.ollama_url.startswith("http") else "http://" + args.ollama_url
        paths = [Path(p) for p in args.paths] or sorted(Path("transcripts").glob("*.jsonl"))
        import_transcripts(mem, paths, ollama_chat_fn(url, args.model))
    elif args.cmd == "export":
        out = [f"# qwen_live memory ({mem.path.resolve()})", "", "## Memories", "",
               "Edit with `memory.py edit ID \"new text\"`, remove with `memory.py forget ID`.", ""]
        for r in mem.list():
            out.append(f"- **#{r['id']}** [{r['kind']}, importance {r['importance']}] {r['text']}  \n  _updated {r['updated']}, from {r['source']}, used {r['uses']}x_")
        out += ["", "## Conversations", ""]
        for r in mem.sessions():
            out += [f"### {r['id']} ({r['turns']} turns)", "", r["summary"] or "_not summarized yet_", ""]
            for e in mem.session_exchanges(r["id"]):
                out += [f"> **You:** {e['user']}", ">", f"> **Qwen:** {e['assistant']}", ""]
        Path(args.file).write_text("\n".join(out) + "\n")
        print(f"wrote {args.file}")
    elif args.cmd == "stats":
        n = lambda t: mem.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"{n('memories')} memories, {n('sessions')} sessions, {n('exchanges')} exchanges in {mem.path}")


if __name__ == "__main__":
    main()
