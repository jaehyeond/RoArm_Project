#!/bin/bash
# File Ownership Check Hook for RoArm Agent Team
# Validates that agents only write to files they own
# Usage: bash file-ownership-check.sh <agent-name>
# Input: JSON via stdin with tool_input.file_path

AGENT_NAME="$1"

INPUT=$(cat)
if [ -z "$INPUT" ]; then
    exit 0
fi

FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)
if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Extract filename
FILE_NAME=$(basename "$FILE_PATH")

ALLOWED=false

case "$AGENT_NAME" in
    "data-agent")
        # data_*.py files and collect_data_manual.py
        if [[ "$FILE_NAME" =~ ^data_ ]] || [[ "$FILE_NAME" == "collect_data_manual.py" ]]; then
            ALLOWED=true
        fi
        ;;
    "pipeline-agent")
        # train_*.py files, run_official_train.py, test_inference_official.py
        if [[ "$FILE_NAME" =~ ^train_ ]] || [[ "$FILE_NAME" == "run_official_train.py" ]] || [[ "$FILE_NAME" == "test_inference_official.py" ]]; then
            ALLOWED=true
        fi
        ;;
    "deploy-agent")
        # deploy_*.py files
        if [[ "$FILE_NAME" =~ ^deploy_ ]]; then
            ALLOWED=true
        fi
        ;;
esac

# Allow writing to agent memory directories
if [[ "$FILE_PATH" =~ \.claude/agent-memory ]]; then
    ALLOWED=true
fi

if [ "$ALLOWED" = false ]; then
    echo "BLOCKED: $AGENT_NAME cannot write to '$FILE_NAME'. Check file ownership rules in agent definition." >&2
    exit 2
fi

exit 0
