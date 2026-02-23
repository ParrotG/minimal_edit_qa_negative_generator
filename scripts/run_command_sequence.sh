#!/usr/bin/env bash

set -Eeuo pipefail

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_command_sequence.sh --file <commands.txt> [--continue-on-error] [--dry-run]
  bash scripts/run_command_sequence.sh -c "<command 1>" -c "<command 2>" [--continue-on-error] [--dry-run]

Options:
  --file <path>          Read commands from a text file (one command per line).
  -c, --command <cmd>    Add one command directly (repeatable).
  --continue-on-error    Continue running remaining commands if one fails.
  --dry-run              Print commands without executing.
  -h, --help             Show this help message.

Notes:
  - Empty lines and lines starting with '#' are ignored in command files.
  - Commands are executed in order using: bash -lc "<command>".
EOF
}

COMMAND_FILE=""
CONTINUE_ON_ERROR=0
DRY_RUN=0
declare -a COMMANDS=()

while (($# > 0)); do
  case "$1" in
    --file)
      if (($# < 2)); then
        echo "Error: --file requires a path." >&2
        exit 2
      fi
      COMMAND_FILE="$2"
      shift 2
      ;;
    -c|--command)
      if (($# < 2)); then
        echo "Error: $1 requires a command string." >&2
        exit 2
      fi
      COMMANDS+=("$2")
      shift 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      print_usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$COMMAND_FILE" ]]; then
  if [[ ! -f "$COMMAND_FILE" ]]; then
    echo "Error: command file does not exist: $COMMAND_FILE" >&2
    exit 2
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    trimmed="${line#"${line%%[![:space:]]*}"}"
    if [[ -z "$trimmed" || "${trimmed:0:1}" == "#" ]]; then
      continue
    fi
    COMMANDS+=("$line")
  done < "$COMMAND_FILE"
fi

if ((${#COMMANDS[@]} == 0)); then
  echo "Error: no commands provided." >&2
  print_usage >&2
  exit 2
fi

total=${#COMMANDS[@]}
failed=0
declare -a FAILED_INDEXES=()

for ((i = 0; i < total; i++)); do
  idx=$((i + 1))
  cmd="${COMMANDS[$i]}"
  echo "[$idx/$total] $cmd"

  if ((DRY_RUN == 1)); then
    continue
  fi

  set +e
  bash -lc "$cmd"
  status=$?
  set -e

  if ((status != 0)); then
    echo "Command failed with exit code $status at step $idx." >&2
    failed=$((failed + 1))
    FAILED_INDEXES+=("$idx")
    if ((CONTINUE_ON_ERROR == 0)); then
      exit "$status"
    fi
  fi
done

if ((failed > 0)); then
  echo "Finished with $failed failed command(s). Failed step(s): ${FAILED_INDEXES[*]}" >&2
  exit 1
fi

echo "Finished successfully. Executed $total command(s)."
