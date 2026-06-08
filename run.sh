#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7,8}"
export API_KEY="${API_KEY:-EMPTY}"
export BASE_URL="${BASE_URL:-}"
export LOCAL_SERVER_PORT="${LOCAL_SERVER_PORT:-29000}"

INFORMATION_DICT="${INFORMATION_DICT:-./KATE/bfcl_eval/Experience/BFCL_v4_multi_turn_base_training_summary_with_embedding.json}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-Qwen/Qwen3-8B}"
TEST_CATEGORY="multi_turn_miss_param_testing, multi_turn_long_context_testing, multi_turn_base_testing, multi_turn_miss_func_testing" 
MODELS=("Qwen/Qwen3-8B-FC")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_prompt \
        --skip-server-setup \
        --temperature 0 \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_prompt --score-dir result/score_prompt
    
    echo "Finished testing $MODEL."
done


for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_fc \
        --skip-server-setup \
        --temperature 0 \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_fc --score-dir result/score_fc
    
    echo "Finished testing $MODEL."
done




for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default \
        --MA-prompt "default" \
        --skip-server-setup \
        --temperature 0 \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default --score-dir result/score_MA_default
    
    echo "Finished testing $MODEL."
done

# Adding different Knowledge for Enhanced Method
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_summary \
        --skip-server-setup \
        --temperature 0 \
        --MA-prompt "default" \
        --if-enhanced \
        --enhanced-method "summary" \
        --trajectory-retrieval \
        --information-dict "$INFORMATION_DICT" \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_summary --score-dir result/score_MA_default_summary
    # trajectory
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_trajectory \
        --skip-server-setup \
        --temperature 0 \
        --MA-prompt "default" \
        --if-enhanced \
        --enhanced-method "trajectory" \
        --trajectory-retrieval \
        --information-dict "$INFORMATION_DICT" \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_trajectory --score-dir result/score_MA_default_trajectory
    
    echo "Finished testing $MODEL."
done


for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_intent \
        --skip-server-setup \
        --temperature 0 \
        --MA-prompt "default" \
        --if-enhanced \
        --enhanced-method "intent" \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_intent --score-dir result/score_MA_default_intent
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_intent_summary \
        --skip-server-setup \
        --temperature 0 \
        --MA-prompt "default" \
        --if-enhanced \
        --enhanced-method "intent_summary" \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_intent_summary --score-dir result/score_MA_default_intent_summary
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_trajectory_summary \
        --skip-server-setup \
        --temperature 0 \
        --MA-prompt "default" \
        --if-enhanced \
        --enhanced-method "trajectory,summary" \
        --trajectory-retrieval \
        --information-dict "$INFORMATION_DICT" \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_trajectory_summary --score-dir result/score_MA_default_trajectory_summary
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_all \
        --skip-server-setup \
        --temperature 0 \
        --MA-prompt "default" \
        --if-enhanced \
        --enhanced-method "trajectory,summary,intent,intent_summary" \
        --trajectory-retrieval \
        --information-dict "$INFORMATION_DICT" \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_all --score-dir result/score_MA_default_all
    
    echo "Finished testing $MODEL."
done

# The knowledge prompt method
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --skip-server-setup \
        --temperature 0 \
        --result-dir result/result_MA_state \
        --MA-prompt "state" \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_state --score-dir result/score_MA_state
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --skip-server-setup \
        --temperature 0 \
        --result-dir result/result_MA_reflection \
        --MA-prompt "reflection" \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_reflection --score-dir result/score_MA_reflection
    
    echo "Finished testing $MODEL."
done


for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --skip-server-setup \
        --temperature 0 \
        --result-dir result/result_MA_intent \
        --MA-prompt "intent" \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_intent --score-dir result/score_MA_intent
    
    echo "Finished testing $MODEL."
done

# Add parallel critic for trajectory and summary

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_same_4_majority \
        --skip-server-setup \
        --num-threads 50 \
        --temperature 0 \
        --sample-num 4 \
        --fusion-method "majority" \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_same_4_majority --score-dir result/score_MA_same_4_majority
    
    echo "Finished testing $MODEL."
done



for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_same_8_majority \
        --fusion-method "majority" \
        --skip-server-setup \
        --num-threads 20 \
        --temperature 0 \
        --sample-num 8 \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_same_8_majority --score-dir result/score_MA_same_8_majority
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_same_16_majority \
        --fusion-method "majority" \
        --num-threads 20 \
        --skip-server-setup \
        --temperature 0 \
        --sample-num 16 \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_same_16_majority --score-dir result/score_MA_same_16_majority
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --reasoning-enhance \
        --result-dir result/result_MA_same_4_reasoning_critic_multi_turn \
        --fusion-prompt "multi_turn" \
        --fusion-method "critic" \
        --skip-server-setup \
        --num-threads 20 \
        --temperature 0 \
        --sample-num 4 \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_same_4_reasoning_critic_multi_turn --score-dir result/score_MA_same_4_reasoning_critic_multi_turn
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --reasoning-enhance \
        --result-dir result/result_MA_same_8_reasoning_critic_multi_turn \
        --fusion-prompt "multi_turn" \
        --fusion-method "critic" \
        --num-threads 20 \
        --skip-server-setup \
        --temperature 0 \
        --sample-num 8 \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_same_8_reasoning_critic_multi_turn --score-dir result/score_MA_same_8_reasoning_critic_multi_turn
    
    echo "Finished testing $MODEL."
done

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --reasoning-enhance \
        --result-dir result/result_MA_same_16_reasoning_critic_multi_turn \
        --fusion-prompt "multi_turn" \
        --fusion-method "critic" \
        --num-threads 20 \
        --skip-server-setup \
        --temperature 0 \
        --sample-num 16 \
        --if-enhanced \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_same_16_reasoning_critic_multi_turn --score-dir result/score_MA_same_16_reasoning_critic_multi_turn
    
    echo "Finished testing $MODEL."
done







for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_trajectory_reasoning_same_4_critic_multi_turn \
        --skip-server-setup \
        --temperature 0 \
        --fusion-method "critic" \
        --fusion-prompt "multi_turn" \
        --reasoning-enhance \
        --num-threads 20 \
        --sample-num 4 \
        --if-enhanced \
        --enhanced-method "trajectory" \
        --trajectory-retrieval \
        --information-dict "$INFORMATION_DICT" \
        --allow-overwrite
    bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_trajectory_reasoning_same_4_critic_multi_turn --score-dir result/score_MA_default_trajectory_reasoning_same_4_critic_multi_turn
    
    echo "Finished testing $MODEL."
done


for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_REPLACED="${MODEL////_}"
    echo "Testing $MODEL..."
    bfcl generate \
        --model "$MODEL" \
        --test-category "$TEST_CATEGORY" \
        --result-dir result/result_MA_default_trajectory_same_4_majority \
        --skip-server-setup \
        --temperature 0 \
        --fusion-method "majority" \
        --MA-prompt "default" \
        --num-threads 20 \
        --sample-num 4 \
        --if-enhanced \
        --enhanced-method "trajectory" \
        --trajectory-retrieval \
        --information-dict "$INFORMATION_DICT" \
        --allow-overwrite
        bfcl evaluate --model "$MODEL" --test-category "$TEST_CATEGORY" --result-dir result/result_MA_default_trajectory_same_4_majority --score-dir result/score_MA_default_trajectory_same_4_majority
    
    echo "Finished testing $MODEL."
done