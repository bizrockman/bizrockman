"""Daily profile status updater for @bizrockman.

Pulls the last 5 public commit messages, asks Claude Haiku for a builder-voice
emoji + badge + Now-block, sets the GitHub user status via GraphQL, and
rewrites the NOW:START/NOW:END region in README.md.

Failure mode: if anything goes sideways, fall back to fire + "Generating Results"
so the profile never ends up bare.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

import anthropic
import requests

GITHUB_USER = os.environ.get("GITHUB_USER", "bizrockman")
PAT = os.environ.get("GH_USER_PAT", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

FALLBACK = {
    "emoji": ":fire:",
    "badge": "Generating Results",
    "now": "Generating Results.",
}

NOW_START = "<!-- NOW:START -->"
NOW_END = "<!-- NOW:END -->"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You produce a GitHub profile status for Danny Gerst (@bizrockman).

His public stance: "Results > Hype". Builder voice. Concrete, present tense.
NEVER use marketing or buzzword language ("leveraging", "next-gen", "empowering",
"infrastructure", "solutions", "capabilities"). Never use exclamation marks.
Never address him in second person.

You receive his most recent public commit messages with repo names, plus how
many days old the latest one is. Your job: distill what he is currently
working on into a tight status that fits on a GitHub profile.

Output STRICT JSON only — no preamble, no code fences, no commentary:
{"emoji": ":<github_shortcode>:", "badge": "<= 60 chars", "now": "<= 280 chars"}

Field rules:
- emoji  : single GitHub shortcode wrapped in colons. Pick from a wide pool;
           rotate naturally by activity. Examples: :fire: :hammer_and_wrench:
           :microscope: :brain: :books: :rocket: :wrench: :test_tube: :gear:
           :pencil2: :sparkles: :package: :zap: :compass: :seedling:
- badge  : what he is building right now. <= 60 characters. One short clause.
           Concrete project or area. No filler.
- now    : 1-3 short sentences. May mention up to two repos by name. May add
           one current curiosity ("reading X on the side"). Markdown OK.
           <= 280 characters. No emoji inside.

Quiet-day handling: if the latest commit is >= 3 days old, switch to a
contemplative tone (thinking / reading / sketching mode) — do NOT pretend
to be coding. Pick a fitting emoji (:brain: :books: :compass: :seedling:).

GOOD examples:
{"emoji":":fire:","badge":"Shipping the OpenChatFormat parser","now":"Wrapping up the OpenChatFormat parser and wiring it into LinkAuth. Reading through DSPy internals on the side."}
{"emoji":":microscope:","badge":"Benchmarking inference stacks","now":"Running AIInferenceBenchmark against three new providers. Notes piling up — write-up next week."}
{"emoji":":brain:","badge":"Thinking, not typing","now":"Quiet stretch on commits — heads-down on NPC Forge design notes and a stack of unpublished papers."}

BAD examples (do not produce):
"Improving authentication infrastructure"
"Leveraging AI capabilities to enable next-gen workflows"
"Working hard on multiple projects!"
"""


BOT_AUTHOR_NAME = "bizrockman-bot"
BOT_MESSAGE_PREFIX = "chore: refresh Now block"


def fetch_recent_commits(limit: int = 5) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if PAT:
        headers["Authorization"] = f"Bearer {PAT}"
    r = requests.get(
        "https://api.github.com/search/commits",
        headers=headers,
        params={
            "q": f"author:{GITHUB_USER}",
            "sort": "author-date",
            "order": "desc",
            "per_page": 30,
        },
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    out: list[dict] = []
    for item in data.get("items", []):
        commit = item.get("commit") or {}
        author = commit.get("author") or {}
        msg = (commit.get("message") or "").splitlines()[0].strip()
        if not msg:
            continue
        if author.get("name") == BOT_AUTHOR_NAME:
            continue
        if msg.startswith(BOT_MESSAGE_PREFIX):
            continue
        low = msg.lower()
        if low.startswith("merge ") or low.startswith("merge pull request"):
            continue
        repo = (item.get("repository") or {}).get("full_name", "")
        date = author.get("date") or (commit.get("committer") or {}).get("date") or ""
        out.append({"repo": repo, "msg": msg, "at": date})
        if len(out) >= limit:
            break
    return out


def ask_claude(commits: list[dict]) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    if commits:
        commit_lines = "\n".join(f"- [{c['repo']}] {c['msg']}" for c in commits)
        latest_iso = commits[0]["at"].replace("Z", "+00:00")
        latest_dt = datetime.fromisoformat(latest_iso)
        age_days = (datetime.now(timezone.utc) - latest_dt).days
        user_msg = (
            f"Latest commit is {age_days} day(s) old.\n"
            f"Recent commits (newest first):\n{commit_lines}"
        )
    else:
        user_msg = (
            "No public commits available right now (none returned by the search). "
            "Do NOT invent a duration like 'three years' or 'months ago' — just produce a "
            "contemplative thinking/reading mode status without specifying any time period."
        )

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
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

    emoji = str(parsed["emoji"]).strip()
    if not (emoji.startswith(":") and emoji.endswith(":")):
        raise ValueError(f"emoji not in :shortcode: form: {emoji!r}")
    badge = str(parsed["badge"]).strip()[:60]
    now = str(parsed["now"]).strip()[:280]
    if not badge or not now:
        raise ValueError("empty badge or now field")
    return {"emoji": emoji, "badge": badge, "now": now}


def set_github_status(emoji: str, message: str) -> None:
    if not PAT:
        raise RuntimeError("GH_USER_PAT not set; cannot update status")
    query = """
    mutation($emoji: String!, $message: String!) {
      changeUserStatus(input: {
        emoji: $emoji,
        message: $message,
        limitedAvailability: false
      }) {
        status { message emoji }
      }
    }
    """
    r = requests.post(
        "https://api.github.com/graphql",
        headers={"Authorization": f"Bearer {PAT}"},
        json={"query": query, "variables": {"emoji": emoji, "message": message}},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")


def update_readme_now(now_text: str, emoji: str) -> None:
    path = "README.md"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bare_emoji = emoji.strip(":")
    pretty_emoji = {
        "fire": "🔥", "hammer_and_wrench": "🛠️", "microscope": "🔬",
        "brain": "🧠", "books": "📚", "rocket": "🚀", "wrench": "🔧",
        "test_tube": "🧪", "gear": "⚙️", "pencil2": "✏️", "sparkles": "✨",
        "package": "📦", "zap": "⚡", "compass": "🧭", "seedling": "🌱",
    }.get(bare_emoji, "🔥")
    block = (
        f"{NOW_START}\n"
        f"> {pretty_emoji} **Now:** {now_text}\n"
        f">\n"
        f"> *Updated: {today}*\n"
        f"{NOW_END}"
    )
    pattern = re.escape(NOW_START) + r".*?" + re.escape(NOW_END)
    new_content, n = re.subn(pattern, block, content, count=1, flags=re.DOTALL)
    if n == 0:
        raise RuntimeError("NOW:START/NOW:END markers not found in README.md")
    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("README updated.")
    else:
        print("README unchanged.")


def write_cache_for_social(commits: list[dict], result: dict, used_fallback: bool) -> None:
    """Persist this run's data so post_to_social.py can read it without re-fetching."""
    import os as _os
    cache_dir = ".state"
    _os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "commits": commits,
        "haiku_result": result,
        "used_fallback": used_fallback,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(f"{cache_dir}/last-haiku.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    result = dict(FALLBACK)
    commits: list[dict] = []
    used_fallback = True
    try:
        commits = fetch_recent_commits()
        print(f"Fetched {len(commits)} commits.", flush=True)
        result = ask_claude(commits)
        used_fallback = False
        print(f"Claude → emoji={result['emoji']} badge={result['badge']!r}", flush=True)
    except Exception as e:
        print(f"[WARN] generation failed, using fallback: {e}", file=sys.stderr, flush=True)
        result = dict(FALLBACK)
        used_fallback = True

    try:
        set_github_status(result["emoji"], result["badge"])
        print("Status set.", flush=True)
    except Exception as e:
        print(f"[WARN] status update failed: {e}", file=sys.stderr, flush=True)
        try:
            set_github_status(FALLBACK["emoji"], FALLBACK["badge"])
            print("Fallback status set.", flush=True)
        except Exception as ee:
            print(f"[ERR] fallback status also failed: {ee}", file=sys.stderr, flush=True)

    try:
        update_readme_now(result["now"], result["emoji"])
    except Exception as e:
        print(f"[ERR] README update failed: {e}", file=sys.stderr, flush=True)
        return 1

    try:
        write_cache_for_social(commits, result, used_fallback)
    except Exception as e:
        print(f"[WARN] cache write failed (social step will be skipped): {e}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
