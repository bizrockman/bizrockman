"""Post a Sonnet-curated update to Bluesky and append it to SHIPPING.md.

Runs after update_status.py. Reads:
  - .state/last-haiku.json  (this run's tagline + commits)
  - .state/social.json       (last posted commit SHA + last style)

Decisions made by Claude Sonnet (more capable than Haiku for editorial
judgment): post or skip, what style (A bullet / B pain-hook / sober /
thread), strict alternation between A and B with content-fit override.

If post: posts to Bluesky, prepends entry to SHIPPING.md, updates state.
If skip: only logs the rationale.

Designed to fail soft — if Sonnet, Bluesky, or anything else hiccups,
the workflow continues and the README/status step results stay intact.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BLUESKY_HANDLE = os.environ.get("BLUESKY_HANDLE", "")
BLUESKY_APP_PASSWORD = os.environ.get("BLUESKY_APP_PASSWORD", "")

CLAUDE_MODEL = "claude-sonnet-4-6"

LAST_HAIKU_PATH = Path(".state/last-haiku.json")
SOCIAL_STATE_PATH = Path(".state/social.json")
SHIPPING_LOG_PATH = Path("SHIPPING.md")

LOG_START = "<!-- LOG:START -->"
LOG_END = "<!-- LOG:END -->"


SYSTEM_PROMPT = """You write Bluesky posts for Danny Gerst (@bizrockman) about his shipping work.

Brand voice: "Results > Hype". Direct-response style à la Jim Edwards — hook-led,
concrete, no marketing tropes. BANNED phrasings: "Just shipped", "thrilled to
announce", "game-changing", "revolutionary", "excited to share", exclamation
marks at sentence end, hashtag walls, emoji as attention-anchor at line start,
"we are pleased to announce".

You receive:
- A `tagline` already produced by Haiku for the GitHub status (builder voice, accurate)
- The raw commit messages with repo names that informed it
- The `last_style` used (so we can alternate A and B when content allows)

Your job: decide whether this is post-worthy, what shape the post takes, and
produce the copy.

Output STRICT JSON only — no preamble, no fences, no commentary:
{"action": "post"|"skip", "style": "A"|"B"|"sober"|"thread", "content": "<text or array>", "rationale": "<one short sentence>"}

Style definitions:
- A: Problem-hook + 2-4 bullet list. Two-line tension setup, then bullets, then
     close. Use when multiple concrete deliverables share one theme.
- B: Pain-scenario hook. Short scenario the reader recognizes ("Your X.
     Your Y."), resolution, close. Use when ONE thing was shipped that solves
     a recognizable pain.
- sober: One or two factual sentences, no hook, no hype. Use when commits are
         real but small (single deliverable without obvious pain framing,
         small refactor with measurable effect).
- thread: JSON array of strings, each <= 300 chars, numbered (1/N). Only when
          4+ substantively different deliverables that don't compress into a list.

Routing rules:
1. Only chore/typo/dep-bump/lint commits → action=skip (timeline noise is the enemy).
2. Single substantive commit → style B unless last_style was B (then A).
3. 2-3 substantive commits with one theme → style A.
4. 4+ unrelated big features → style thread.
5. Mixed-bag small but real work → style sober.
6. Otherwise: prefer alternation against last_style.

Hard constraints:
- <= 300 chars per post (Bluesky limit). Each thread post also <= 300.
- Include the most relevant repository URL once at the end of the post (or last
  thread post). Format: https://github.com/ORG/REPO. Pick the repo that the
  most commits target. Skip the URL only if no obvious repo applies.
- Don't invent quantities (commits, lines, features) not in the source.
- Don't echo specific numbers from commit messages verbatim if they may be stale.
- Don't address the reader as "you" more than once per post.
- Style A bullet list: 2-4 bullets, never just 1 (looks awkward).
- The tagline is the editorial seed but you are not bound to its exact wording.

Few-shot calibration (the voice you should produce):

Style B example:
"Your Claude conversation hits the budget wall. Cursor doesn't know what you'd been building.

ocf-py fixes that today: exporters for Claude Code and Cursor, plus a Markdown renderer.

https://github.com/open-conversation-format/ocf-py"

Style A example:
"Every AI tool stores conversations in its own silo.
Three pieces of ocf-py now address that:

- Claude Code exporter (redaction built in)
- Cursor exporter
- Markdown renderer

https://github.com/open-conversation-format/ocf-py"

Sober example:
"Cleaned up the OpenXE customer interface adapter — the legacy XML parser is gone, the new schema validator is in. Quiet work that paid down a year of debt.

https://github.com/bizrockman/OpenXE"
"""


def load_haiku_output() -> dict | None:
    if not LAST_HAIKU_PATH.exists():
        return None
    try:
        return json.loads(LAST_HAIKU_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] could not read haiku cache: {e}", file=sys.stderr, flush=True)
        return None


def load_social_state() -> dict:
    if SOCIAL_STATE_PATH.exists():
        try:
            return json.loads(SOCIAL_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_posted_sha": None, "last_style": None, "last_posted_at": None}


def save_social_state(state: dict) -> None:
    SOCIAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCIAL_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def latest_commit_signature(commits: list[dict]) -> str:
    """Stable identity for the most recent commit, used for dedup."""
    if not commits:
        return ""
    top = commits[0]
    return f"{top.get('repo', '')}::{top.get('msg', '')}::{top.get('at', '')}"


def ask_sonnet(haiku: dict, commits: list[dict], last_style: str | None) -> dict:
    if not ANTHROPIC_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    tagline = (haiku.get("haiku_result") or {}).get("now", "")
    commit_lines = "\n".join(f"- [{c.get('repo', '')}] {c.get('msg', '')}" for c in commits)

    user_msg = (
        f'tagline: "{tagline}"\n\n'
        f"recent commits (newest first):\n{commit_lines}\n\n"
        f'last_style: "{last_style or "B"}"'
    )

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    parsed = json.loads(raw)

    action = parsed.get("action")
    if action not in ("post", "skip"):
        raise ValueError(f"invalid action: {action!r}")
    if action == "post":
        if not parsed.get("content"):
            raise ValueError("post action without content")
    return parsed


URL_RE = re.compile(r"https?://[^\s]+")


def post_to_bluesky_single(text: str) -> str:
    """Post a single skeet with rich-text facets for embedded URLs. Returns post URI."""
    from atproto import Client, client_utils

    if not (BLUESKY_HANDLE and BLUESKY_APP_PASSWORD):
        raise RuntimeError("Bluesky credentials not set")

    client = Client()
    client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)

    tb = client_utils.TextBuilder()
    last_end = 0
    for m in URL_RE.finditer(text):
        if m.start() > last_end:
            tb.text(text[last_end : m.start()])
        url = m.group(0).rstrip(".,;:!?)")
        tb.link(url, url)
        # any trailing punctuation we stripped
        tail = m.group(0)[len(url):]
        if tail:
            tb.text(tail)
        last_end = m.end()
    if last_end < len(text):
        tb.text(text[last_end:])

    response = client.send_post(text=tb.build_text(), facets=tb.build_facets())
    return getattr(response, "uri", "")


def post_to_bluesky_thread(parts: list[str]) -> list[str]:
    """Post a thread (root + replies). Returns list of URIs in order."""
    from atproto import Client, client_utils, models

    if not (BLUESKY_HANDLE and BLUESKY_APP_PASSWORD):
        raise RuntimeError("Bluesky credentials not set")
    if not parts:
        return []

    client = Client()
    client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)

    def build_tb(text: str):
        tb = client_utils.TextBuilder()
        last_end = 0
        for m in URL_RE.finditer(text):
            if m.start() > last_end:
                tb.text(text[last_end : m.start()])
            url = m.group(0).rstrip(".,;:!?)")
            tb.link(url, url)
            tail = m.group(0)[len(url):]
            if tail:
                tb.text(tail)
            last_end = m.end()
        if last_end < len(text):
            tb.text(text[last_end:])
        return tb

    uris: list[str] = []
    root_ref = None
    parent_ref = None
    for i, part in enumerate(parts):
        tb = build_tb(part)
        if i == 0:
            resp = client.send_post(text=tb.build_text(), facets=tb.build_facets())
            root_ref = models.create_strong_ref(resp)
            parent_ref = root_ref
        else:
            reply_ref = models.AppBskyFeedPost.ReplyRef(parent=parent_ref, root=root_ref)
            resp = client.send_post(
                text=tb.build_text(), facets=tb.build_facets(), reply_to=reply_ref
            )
            parent_ref = models.create_strong_ref(resp)
        uris.append(getattr(resp, "uri", ""))
    return uris


def append_to_shipping_log(content: str | list[str], style: str, posted_uris: list[str]) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = content if isinstance(content, str) else "\n\n".join(
        f"**{i+1}/{len(content)}** {p}" for i, p in enumerate(content)
    )
    posted_line = ""
    if posted_uris:
        # Bluesky URI format: at://did:plc:.../app.bsky.feed.post/<rkey>
        # Convert to web URL
        web_urls = []
        for uri in posted_uris:
            m = re.match(r"at://([^/]+)/app\.bsky\.feed\.post/(.+)", uri)
            if m and BLUESKY_HANDLE:
                web_urls.append(
                    f"https://bsky.app/profile/{BLUESKY_HANDLE}/post/{m.group(2)}"
                )
        if web_urls:
            posted_line = f"\n*Posted to Bluesky: [{web_urls[0]}]({web_urls[0]})*\n"

    entry = (
        f"### {today} · style: {style}\n"
        f"\n"
        f"{body}\n"
        f"{posted_line}"
        f"\n"
        f"---\n"
        f"\n"
    )

    if SHIPPING_LOG_PATH.exists():
        existing = SHIPPING_LOG_PATH.read_text(encoding="utf-8")
        if LOG_START in existing and LOG_END in existing:
            # insert immediately after LOG:START so newest is on top
            new_existing = existing.replace(
                LOG_START, f"{LOG_START}\n\n{entry.rstrip()}", 1
            )
            SHIPPING_LOG_PATH.write_text(new_existing, encoding="utf-8")
            return
        # markers missing — append at end
        SHIPPING_LOG_PATH.write_text(existing + "\n" + entry, encoding="utf-8")
    else:
        header = (
            "# Shipping Log\n\n"
            "What got built and shipped, in reverse-chronological order. "
            "Auto-generated by the daily status workflow — every entry is a real "
            "commit-day distilled by the same pipeline that updates the GitHub "
            "status badge.\n\n"
            f"{LOG_START}\n\n"
            f"{entry.rstrip()}\n\n"
            f"{LOG_END}\n"
        )
        SHIPPING_LOG_PATH.write_text(header, encoding="utf-8")


def main() -> int:
    haiku = load_haiku_output()
    if not haiku:
        print("[INFO] no haiku cache; skipping social step.", flush=True)
        return 0

    if haiku.get("used_fallback"):
        print("[INFO] haiku run used fallback (no real commits); skipping social.", flush=True)
        return 0

    commits = haiku.get("commits") or []
    if not commits:
        print("[INFO] no commits in cache; skipping social.", flush=True)
        return 0

    state = load_social_state()
    sig = latest_commit_signature(commits)
    if sig and sig == state.get("last_posted_sha"):
        print(f"[INFO] no new commits since last post; skipping.", flush=True)
        return 0

    try:
        decision = ask_sonnet(haiku, commits, state.get("last_style"))
    except Exception as e:
        print(f"[ERR] Sonnet call failed: {e}", file=sys.stderr, flush=True)
        return 1

    rationale = decision.get("rationale", "")
    print(
        f"Sonnet → action={decision['action']} style={decision.get('style')} "
        f"rationale={rationale!r}",
        flush=True,
    )

    if decision["action"] == "skip":
        print("[INFO] Sonnet decided to skip — nothing to post.", flush=True)
        # don't update state.last_posted_sha; we may post a future bundle that
        # includes this content. But DO record the commit sig as "considered"
        # to avoid repeated Sonnet calls on the same data.
        state["last_considered_sha"] = sig
        save_social_state(state)
        return 0

    content = decision["content"]
    style = decision.get("style", "sober")

    posted_uris: list[str] = []
    try:
        if isinstance(content, list) and style == "thread":
            posted_uris = post_to_bluesky_thread(content)
            print(f"Bluesky thread posted: {len(posted_uris)} parts.", flush=True)
        else:
            text = content if isinstance(content, str) else "\n\n".join(content)
            uri = post_to_bluesky_single(text)
            posted_uris = [uri]
            print(f"Bluesky post created: {uri}", flush=True)
    except Exception as e:
        print(f"[ERR] Bluesky post failed: {e}", file=sys.stderr, flush=True)
        return 1

    try:
        append_to_shipping_log(content, style, posted_uris)
        print("SHIPPING.md updated.", flush=True)
    except Exception as e:
        print(f"[WARN] shipping log update failed: {e}", file=sys.stderr, flush=True)
        # post is already on Bluesky — don't fail the whole job

    state["last_posted_sha"] = sig
    if style in ("A", "B"):
        state["last_style"] = style
    state["last_posted_at"] = datetime.now(timezone.utc).isoformat()
    save_social_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
