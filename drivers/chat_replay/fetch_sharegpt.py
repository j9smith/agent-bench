"""Pull a ShareGPT-format conversation set and normalise it to data/sharegpt.json.

Tries a couple of well-known mirrors; if the network or the hub is
uncooperative, point --local at any JSON file with either shape:
    [{"conversations": [{"from": "human", "value": "..."}, ...]}, ...]
    [{"messages":      [{"role": "user",  "content": "..."}, ...]}, ...]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = [
    ("anon8231489123/ShareGPT_Vicuna_unfiltered", None),
    ("lmsys/lmsys-chat-1m", "train"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sharegpt.json")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--local", type=Path, help="normalise an existing local JSON instead")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.local:
        rows = json.loads(args.local.read_text())[: args.limit]
        args.out.write_text(json.dumps(rows))
        print(f"wrote {len(rows)} conversations -> {args.out}")
        return 0

    from datasets import load_dataset

    for repo, split in CANDIDATES:
        try:
            ds = load_dataset(repo, split=split or "train")
            rows = []
            for r in ds:
                conv = r.get("conversations") or r.get("conversation")
                if conv:
                    rows.append({"conversations": list(conv)})
                if len(rows) >= args.limit:
                    break
            if rows:
                args.out.write_text(json.dumps(rows))
                print(f"wrote {len(rows)} conversations from {repo} -> {args.out}")
                return 0
        except Exception as exc:
            print(f"[warn] {repo}: {exc}", file=sys.stderr)

    print("could not fetch any dataset; use --local", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
