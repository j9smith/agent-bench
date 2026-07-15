#!/usr/bin/env bash
set -euo pipefail
exec python -m drivers.openhands.run_openhands "$@"
