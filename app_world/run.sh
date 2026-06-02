#!/bin/bash

usage() {
    cat << EOF
用法: $0 [选项]

  -d, --dataset DATASET      test_normal|test_challenge|both (默认: test_normal)
  -p, --parallel NUM         并行数 (默认: 1)
  -r, --retrieval            启用检索增强
  --restart                  重启实验
  --memp                     使用 memp.py
  --test2                    使用 appworld_test_2.py
  --gpt4                     GPT-4.1
  --qwen32b                  Qwen3-32B (默认)
  --qwen8b                   Qwen3-8B
  -h, --help                 帮助

示例: $0 --gpt4 -r -p 4 -d both
EOF
    exit 1
}

# 默认值
MODEL="Qwen/Qwen3-32B"
BASE_URL="http://175.102.130.120:28000/v1"
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
        --gpt4) MODEL="gpt-4.1-2025-04-14"; BASE_URL="https://api.v3.cm/v1";
                KEY="${OPENAI_API_KEY:-}"; shift;;
        --qwen32b) MODEL="Qwen/Qwen3-32B"; BASE_URL="http://175.102.130.120:28000/v1"; KEY="EMPTY"; shift;;
        --qwen8b) MODEL="Qwen/Qwen3-8B"; BASE_URL="http://210.75.240.154:28001/v1"; KEY="EMPTY"; shift;;
        -h|--help) usage;;
        *) echo "未知选项: $1"; usage;;
    esac
done

export BASE_URL="$BASE_URL" KEY="$KEY" MODEL="$MODEL"

# 提取模型简称
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

    [ $? -eq 0 ] && appworld evaluate "$exp" "$ds" || { echo "实验失败"; return 1; }
}

case "$DATASET" in
    both) run_experiment "test_normal" && run_experiment "test_challenge";;
    test_normal|test_challenge) run_experiment "$DATASET";;
    *) echo "错误: 数据集须为 test_normal, test_challenge 或 both"; exit 1;;
esac
