#!/usr/bin/env bash
set -euo pipefail
exec python -m drivers.chat_replay.replay "$@"
