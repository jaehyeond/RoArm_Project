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

# Fail-closed: if python3 is not available, block everything
if ! command -v python3 &>/dev/null; then
    echo "BLOCKED: python3 not found. Cannot parse hook input safely." >&2
    exit 2
fi

FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Extract filename
FILE_NAME=$(basename "$FILE_PATH")

ALLOWED=false

case "$AGENT_NAME" in
    # === Engineering Workers ===
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
    # === Team A: Robotics ===
    "robotics-manipulation")
        # trajectory_*.py files
        if [[ "$FILE_NAME" =~ ^trajectory_ ]]; then
            ALLOWED=true
        fi
        ;;
    "robotics-sim2real")
        # sim_*.py files
        if [[ "$FILE_NAME" =~ ^sim_ ]]; then
            ALLOWED=true
        fi
        ;;
    "robotics-hardware")
        # hw_*.py and calibrate_*.py files
        if [[ "$FILE_NAME" =~ ^hw_ ]] || [[ "$FILE_NAME" =~ ^calibrate_ ]]; then
            ALLOWED=true
        fi
        ;;
    # === Team B: Physical AI ===
    "pai-vla-model")
        # model_*.py files
        if [[ "$FILE_NAME" =~ ^model_ ]]; then
            ALLOWED=true
        fi
        ;;
    "pai-data-efficiency")
        # augment_*.py and self_improve_*.py files
        if [[ "$FILE_NAME" =~ ^augment_ ]] || [[ "$FILE_NAME" =~ ^self_improve_ ]]; then
            ALLOWED=true
        fi
        ;;
    "pai-deployment")
        # monitor_*.py and safety_*.py files
        if [[ "$FILE_NAME" =~ ^monitor_ ]] || [[ "$FILE_NAME" =~ ^safety_ ]]; then
            ALLOWED=true
        fi
        ;;
    # === Team C: Research Methods ===
    "research-experiment")
        # experiment_*.py and eval_*.py files
        if [[ "$FILE_NAME" =~ ^experiment_ ]] || [[ "$FILE_NAME" =~ ^eval_ ]]; then
            ALLOWED=true
        fi
        ;;
    "research-analysis")
        # analysis_*.py and figure_*.py files
        if [[ "$FILE_NAME" =~ ^analysis_ ]] || [[ "$FILE_NAME" =~ ^figure_ ]]; then
            ALLOWED=true
        fi
        ;;
    "research-writing")
        # paper/ directory files
        if [[ "$FILE_PATH" =~ /paper/ ]]; then
            ALLOWED=true
        fi
        ;;
    # === Execution Roles (역할 agent 4개, 2026-09-19 등록 — D492) ===
    # 산출 경계 = run output 폴더. 상태 원장은 아래 별도 차단으로 막는다.
    "deme-runner"|"raw-accountant"|"replay-renderer"|"independent-auditor")
        if [[ "$FILE_PATH" =~ /claudedocs/runtime_logs/ ]]; then
            ALLOWED=true
        fi
        ;;
esac

# State-ledger exclusivity: only the main coordinator session writes these.
if [[ "$FILE_PATH" =~ /START_HERE\.md$ ]] \
    || [[ "$FILE_PATH" =~ /claudedocs/(DECISIONS|DECISIONS_ACTIVE|EXPERIMENT_LEDGER|LEDGER_RECENT)\.md$ ]] \
    || [[ "$FILE_PATH" =~ /claudedocs/relay/ ]]; then
    echo "BLOCKED: state ledger '$FILE_NAME' is owned by the main coordinator session only (AGENTS.md exclusivity rule)." >&2
    exit 2
fi

# Allow writing to agent memory directories
if [[ "$FILE_PATH" =~ \.claude/agent-memory ]]; then
    ALLOWED=true
fi

if [ "$ALLOWED" = false ]; then
    echo "BLOCKED: $AGENT_NAME cannot write to '$FILE_NAME'. Check file ownership rules in agent definition." >&2
    exit 2
fi

exit 0
