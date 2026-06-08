#!/bin/bash

usage() {
    cat << EOF
Usage: $0 [options]

  -d, --dataset DATASET      test_normal|test_challenge|both (default: test_normal)
  -p, --parallel NUM         number of parallel workers (default: 1)
  -r, --retrieval            enable retrieval-augmented mode
  --restart                  restart the experiment
  --memp                     use memp.py
  --test2                    use appworld_test_2.py
  --gpt4                     GPT-4.1
  --qwen32b                  Qwen3-32B (default)
  --qwen8b                   Qwen3-8B
  -h, --help                 show this help message

Example: $0 --gpt4 -r -p 4 -d both
EOF
    exit 1
}

# Defaults
MODEL="Qwen/Qwen3-32B"
BASE_URL="${QWEN32B_BASE_URL:-}"
KEY="EMPTY"
DATASET="test_normal"
PARALLEL=1
RETRIEVAL="" RESTART="" USE_MEMP=false USE_TEST2=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset) DATASET="$2"; shift 2;;
        -p|--parallel) PARALLEL="$2"; shift 2;;
        -r|--retrieval) RETRIEVAL="--retrieval_enhance"; shift;;
        --restart) RESTART="--restart"; shift;;
        --memp) USE_MEMP=true; shift;;
        --test2) USE_TEST2=true; shift;;
        --gpt4) MODEL="gpt-4.1-2025-04-14"; BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}";
                KEY="${OPENAI_API_KEY:-}"; shift;;
        --qwen32b) MODEL="Qwen/Qwen3-32B"; BASE_URL="${QWEN32B_BASE_URL:-}"; KEY="EMPTY"; shift;;
        --qwen8b) MODEL="Qwen/Qwen3-8B"; BASE_URL="${QWEN8B_BASE_URL:-}"; KEY="EMPTY"; shift;;
        -h|--help) usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

if [[ -z "$BASE_URL" ]]; then
    echo "Error: BASE_URL is not set. Export QWEN32B_BASE_URL, QWEN8B_BASE_URL, or OPENAI_BASE_URL before running."
    exit 1
fi

export BASE_URL="$BASE_URL" KEY="$KEY" MODEL="$MODEL"

# Extract short model name
case "$MODEL" in
    *gpt-4*) model_short="gpt-4.1-2025-04-14";;
    *Qwen3-32B*) model_short="Qwen3-32B";;
    *Qwen3-8B*) model_short="Qwen3-8B";;
    *) model_short=$(basename "$MODEL");;
esac

run_experiment() {
    local ds=$1
    local ret_suffix=$([[ -n "$RETRIEVAL" ]] && echo "True" || echo "False")

    if $USE_MEMP; then
        local exp="memp_${ds}_${model_short}"
        python memp.py --dataset_name "$ds" --experiment_name "$exp" $RETRIEVAL $RESTART
    elif $USE_TEST2; then
        local exp="m_${ds}_${model_short}_retrieval_${ret_suffix}_${PARALLEL}"
        python appworld_test_2.py --dataset_name "$ds" --experiment_name "$exp" --parallel_decode "$PARALLEL" $RETRIEVAL $RESTART
    else
        local exp="${ds}_${model_short}_retrieval_${ret_suffix}_${PARALLEL}"
        python appworld_test.py --dataset_name "$ds" --experiment_name "$exp" --parallel_decode "$PARALLEL" $RETRIEVAL $RESTART
    fi

    [ $? -eq 0 ] && appworld evaluate "$exp" "$ds" || { echo "Experiment failed"; return 1; }
}

case "$DATASET" in
    both) run_experiment "test_normal" && run_experiment "test_challenge";;
    test_normal|test_challenge) run_experiment "$DATASET";;
    *) echo "Error: dataset must be test_normal, test_challenge, or both"; exit 1;;
esac
