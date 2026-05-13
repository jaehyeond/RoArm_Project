#!/usr/bin/env bash
# Launch Codex with a Korean UTF-8 locale and inline TUI mode.
#
# Why:
# - In VSCode integrated terminal + ibus Korean IME, Codex TUI may duplicate
#   committed Hangul syllables in raw/alternate-screen input handling.
# - Claude Code can be fine while Codex CLI duplicates text, so keep the workaround
#   scoped to Codex only.
#
# Usage:
#   ./codex-ko.sh
#   ./codex-ko.sh resume --last
#   ./codex-ko.sh -C /path/to/project

set -euo pipefail

export LANG=ko_KR.UTF-8
export LANGUAGE=ko:en
export LC_ALL=ko_KR.UTF-8
export LC_CTYPE=ko_KR.UTF-8

exec codex --no-alt-screen "$@"
