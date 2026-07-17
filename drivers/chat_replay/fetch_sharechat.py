"""Fetch ShareChat and prepare it for gap-accurate replay.

Why ShareChat over ShareGPT: it carries REAL per-message timestamps for the GPT
and Grok subsets (message_create_time), so we can replay with the actual human
inter-turn gaps instead of a fixed/zero delay. That turns chat gap-duration from a
replay artifact into a measured quantity -- which matters because the whole
retention argument is about gap observability.

We keep ONLY the timestamped platforms (GPT, Grok = ~90% of turns). The other three
(Claude, Gemini, Perplexity) have no per-message timing and would reintroduce the
zero-gap artifact, so they're dropped.

Schema (from the ShareChat release, one row per MESSAGE):
    platform, url, turns_count, message_index, role, plain_text, detected_language_final
    GPT/Grok extra: message_create_time (per-message epoch)

Output: data/sharechat.json -- a list of conversations, each:
    {"platform","url","turns":[{"role","content","t"}], "gaps":[...]}
  where gaps[k] = seconds between user-turn k's arrival and the prior assistant
  reply completing (derived from message_create_time). Replayer sleeps these
  (capped) between turns.

NOTE: HuggingFace host must be reachable (it is on the pod; it is NOT in the build
sandbox, so this script is verified against a synthetic fixture, then run for real
on the pod). If the schema drifted since this was written, the field-name constants
below are the only thing to adjust.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- schema field names (adjust here if the release schema drifts) ---
F_PLATFORM = "platform"
F_URL = "url"
F_IDX = "message_index"
F_ROLE = "role"
F_TEXT = "plain_text"
F_TS = "message_create_time"          # GPT + Grok per-message timestamp
KEEP_PLATFORMS = {"GPT", "Grok"}      # the timestamped subset
ROLE_MAP = {"user": "user", "human": "user", "assistant": "assistant",
            "gpt": "assistant", "bot": "assistant", "model": "assistant"}


def _parse_ts(v):
    """message_create_time may be epoch seconds (float) or ISO string. Return float
    epoch seconds, or None if unparseable."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def conversations_from_rows(rows) -> list[dict]:
    """Group message-rows into conversations, keeping timestamped platforms only.

    `rows` is any iterable of dict-like message records (HF dataset rows, or the
    test fixture). Grouped by url, ordered by message_index.
    """
    by_url: dict[str, list[dict]] = {}
    for r in rows:
        if r.get(F_PLATFORM) not in KEEP_PLATFORMS:
            continue
        url = r.get(F_URL)
        if not url:
            continue
        by_url.setdefault(url, []).append(r)

    convs = []
    for url, msgs in by_url.items():
        msgs.sort(key=lambda m: m.get(F_IDX, 0))
        turns = []
        for m in msgs:
            role = ROLE_MAP.get(str(m.get(F_ROLE, "")).lower())
            content = m.get(F_TEXT) or ""
            if not role or not content:
                continue
            turns.append({"role": role, "content": content,
                          "t": _parse_ts(m.get(F_TS))})
        if sum(1 for t in turns if t["role"] == "user") < 1:
            continue

        # Derive per-user-turn gaps: time between the previous message's timestamp
        # and this user turn's timestamp. That's the human think/away time the
        # server can't see. None where timestamps are missing.
        gaps = []
        for i, t in enumerate(turns):
            if t["role"] != "user":
                continue
            if i == 0 or t["t"] is None or turns[i - 1]["t"] is None:
                gaps.append(None)
            else:
                gaps.append(max(0.0, turns[i]["t"] - turns[i - 1]["t"]))

        convs.append({"platform": msgs[0].get(F_PLATFORM), "url": url,
                      "turns": turns, "gaps": gaps})
    return convs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sharechat.json")
    ap.add_argument("--dataset", default="tucnguyen/ShareChat")
    ap.add_argument("--split", default="train")
    ap.add_argument("--configs", default="chatgpt,grok",
                    help="comma list of platform configs to load (timestamped ones only)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max conversations to keep (0 = all)")
    ap.add_argument("--min-turns", type=int, default=2)
    args = ap.parse_args()

    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        print("pip install datasets", file=sys.stderr)
        return 2

    # ShareChat is split into PER-PLATFORM configs (chatgpt, grok, claude, gemini,
    # perplexity), not one table with a platform column. Only chatgpt + grok carry
    # per-message timestamps, so we load only those and concatenate.
    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    try:
        available = get_dataset_config_names(args.dataset)
        wanted = [c for c in wanted if c in available]
        if not wanted:
            print(f"[sharechat] none of the requested configs exist. available: "
                  f"{available}", file=sys.stderr)
            return 2
    except Exception:
        pass  # if listing fails, just try to load what was asked

    all_rows = []
    for cfg in wanted:
        print(f"[sharechat] loading config '{cfg}'...", file=sys.stderr)
        d = load_dataset(args.dataset, cfg, split=args.split)
        # tag each row's platform from the config, so conversations_from_rows can keep
        # it. The config name IS the platform here.
        plat = {"chatgpt": "GPT", "grok": "Grok"}.get(cfg, cfg)
        for r in d:
            row = dict(r)
            row[F_PLATFORM] = plat
            all_rows.append(row)

    print(f"[sharechat] {len(all_rows)} message rows across {wanted}; "
          f"grouping...", file=sys.stderr)
    convs = conversations_from_rows(all_rows)
    convs = [c for c in convs
             if sum(1 for t in c["turns"] if t["role"] == "user") >= args.min_turns]
    if args.limit > 0:
        convs = convs[:args.limit]

    # quick gap stats so you can pick a sensible --gap-cap at replay time
    all_gaps = [g for c in convs for g in c["gaps"] if g is not None]
    all_gaps.sort()
    def pct(p):
        return all_gaps[int(p * (len(all_gaps) - 1))] if all_gaps else float("nan")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(convs))
    print(f"[sharechat] wrote {len(convs)} conversations -> {args.out}", file=sys.stderr)
    if all_gaps:
        print(f"[sharechat] inter-turn gap seconds: median={pct(.5):.0f} "
              f"p90={pct(.9):.0f} p99={pct(.99):.0f} max={all_gaps[-1]:.0f} "
              f"(n={len(all_gaps)}). Pick --gap-cap accordingly.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
