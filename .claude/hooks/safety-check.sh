#!/bin/bash
# Safety Check Hook for RoArm Agent Team
# Blocks: git commands, robot hardware commands, dangerous operations
# Input: JSON via stdin with tool_input.command

INPUT=$(cat)
if [ -z "$INPUT" ]; then
    exit 0
fi

# Fail-closed: if python3 is not available, block everything
if ! command -v python3 &>/dev/null; then
    echo "BLOCKED: python3 not found. Cannot parse hook input safely." >&2
    exit 2
fi

COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$COMMAND" ]; then
    exit 0
fi

# Block git commands (Lead only)
if echo "$COMMAND" | grep -qP '\bgit\s+(push|pull|commit|add|checkout|branch|merge|rebase|reset|stash|tag|remote|fetch|clone|init)\b'; then
    echo "BLOCKED: git commands are restricted to Lead agent only. Report this need to Lead." >&2
    exit 2
fi

# Block direct robot serial commands (Lead approval required)
if echo "$COMMAND" | grep -qP 'serial\.Serial|/dev/ttyUSB|torque_set|joints_angle_ctrl|move_init|T:106'; then
    echo "BLOCKED: Direct robot hardware commands require Lead approval. Design the code but do not execute." >&2
    exit 2
fi

# Block dangerous file operations
if echo "$COMMAND" | grep -qP '\brm\s+-rf\b'; then
    echo "BLOCKED: Recursive delete operations not allowed. Use targeted file operations." >&2
    exit 2
fi

# Block lerobot-train execution (Pipeline Agent designs config, Lead approves execution)
if echo "$COMMAND" | grep -qP 'lerobot-train|lerobot\.scripts\.train'; then
    echo "BLOCKED: Training execution requires Lead approval. Design the config and report back." >&2
    exit 2
fi

exit 0
