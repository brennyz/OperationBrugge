#!/usr/bin/env python3
"""Generate one-off MP3 narrations for the hidden Operation Brugge audio tour.

Reads tour/stops.json and writes audio/<id>.mp3 plus audio/manifest.json.
Skips a stop when the MP3 already exists and the text+model hash is unchanged,
unless you pass --force.

The Deepgram key must come from the environment (or a local .env file).
Never commit the key, a .env file, or any *.key file.

Default voice: flux-lotte-nl (Flemish / Belgian Dutch) via POST /v2/speak.
Aura-2 on /v1/speak only has Netherlands Dutch (nl-nl), no Flemish accent.

Usage, from the repo root:

    export DEEPGRAM_API_KEY='...'     # or put it in .env
    python3 tools/gen_audio.py

    DEEPGRAM_MODEL=flux-dirk-nl python3 tools/gen_audio.py
    python3 tools/gen_audio.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOPS_PATH = ROOT / "tour" / "stops.json"
AUDIO_DIR = ROOT / "audio"
MANIFEST_PATH = AUDIO_DIR / "manifest.json"

# Flux Flemish (Belgian Dutch) voices live on the v2 batch REST endpoint.
# Aura-2 Dutch voices (aura-2-*-nl) are Netherlands Dutch only, on /v1/speak.
DEFAULT_MODEL = "flux-lotte-nl"
FLUX_URL = "https://api.deepgram.com/v2/speak"
AURA_URL = "https://api.deepgram.com/v1/speak"
FLEMISH_MODELS = (
    "flux-lotte-nl",
    "flux-veerle-nl",
    "flux-dirk-nl",
    "flux-niels-nl",
)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_stops() -> list[dict]:
    data = json.loads(STOPS_PATH.read_text(encoding="utf-8"))
    stops = data.get("stops") or []
    if not stops:
        sys.exit(f"No stops found in {STOPS_PATH}")
    for s in stops:
        if not s.get("id") or not s.get("text"):
            sys.exit(f"Each stop needs an id and text: {s!r}")
    return stops


def load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        return {"model": None, "stops": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"model": None, "stops": {}}


def save_manifest(manifest: dict) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def endpoint_for(model: str) -> str:
    if model.startswith("flux-"):
        return FLUX_URL
    return AURA_URL


def speak(text: str, key: str, model: str) -> bytes:
    params = urllib.parse.urlencode({"model": model, "encoding": "mp3"})
    url = f"{endpoint_for(model)}?{params}"
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", "replace")
        raise SystemExit(
            f"Deepgram HTTP {exc.code} for model {model}: {err[:800]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach Deepgram: {exc}") from exc
    if len(data) < 100 or "json" in ctype.lower() or "text" in ctype.lower():
        raise SystemExit(
            f"Unexpected Deepgram response ({ctype}, {len(data)} bytes): "
            f"{data[:400]!r}"
        )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every stop, even if the text is unchanged.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Deepgram model id (default: $DEEPGRAM_MODEL or {DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = (os.environ.get("DEEPGRAM_API_KEY") or "").strip()
    if not key:
        print(
            "No DEEPGRAM_API_KEY found. Export it, or put it in a .env file "
            "in the repo root (that file is gitignored).",
            file=sys.stderr,
        )
        return 1

    model = (args.model or os.environ.get("DEEPGRAM_MODEL") or DEFAULT_MODEL).strip()
    stops = load_stops()
    manifest = load_manifest()
    manifest.setdefault("stops", {})
    manifest["model"] = model
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if model.startswith("flux-") and model in FLEMISH_MODELS:
        print(f"Using Flemish Flux voice {model} via {FLUX_URL}")
    elif model.startswith("aura-") and model.endswith("-nl"):
        print(
            f"Using Aura-2 Dutch voice {model} via {AURA_URL}. "
            "This is Netherlands Dutch, not Flemish."
        )
    else:
        print(f"Using model {model} via {endpoint_for(model)}")

    made = skipped = 0
    for stop in stops:
        sid = stop["id"]
        text = stop["text"].strip()
        digest = sha256_hex(model + "\n" + text)
        out = AUDIO_DIR / f"{sid}.mp3"
        prev = manifest["stops"].get(sid) or {}
        if (
            not args.force
            and out.is_file()
            and out.stat().st_size > 100
            and prev.get("hash") == digest
        ):
            print(f"skip  {sid} (unchanged)")
            skipped += 1
            continue
        print(f"speak {sid} ...")
        audio = speak(text, key, model)
        out.write_bytes(audio)
        manifest["stops"][sid] = {
            "hash": digest,
            "bytes": len(audio),
            "model": model,
        }
        save_manifest(manifest)
        print(f"  wrote {out.relative_to(ROOT)} ({len(audio)} bytes)")
        made += 1

    save_manifest(manifest)
    print(f"Done. generated={made} skipped={skipped} total={len(stops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
