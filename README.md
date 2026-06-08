# Pushing the Limits of LLM Tool Calling via Experiential Knowledge Integration and Activation

A comprehensive framework for evaluating and enhancing large language models' tool-calling capabilities using experience retrieval and multi-agent methods.

---

## 📁 Project Structure

```
KATE/
├── bfcl_eval/              # Berkeley Function Calling Leaderboard evaluation framework
│   ├── constants/          # Model configurations and constants
│   ├── data/              # Test datasets and benchmarks
│   ├── eval_checker/      # Evaluation and scoring logic
│   ├── model_handler/     # Model inference handlers
│   │   ├── local_inference/   # Local model handlers (Qwen, Llama, etc.)
│   │   └── api_inference/     # API-based model handlers (GPT, Claude, etc.)
│   └── scripts/           # Utility scripts
|   ├── Experience/            # Knowledge base for experience retrieval
│       ├── *.json           # Experience trajectories and summaries
│       └── *_with_embedding.json  # Experiences with embeddings for retrieval
│
├── app_world/             # AppWorld benchmark integration
│   ├── data/             # AppWorld test data
│   ├── Experience/       # Experience storage for AppWorld
│   └── run.sh           # AppWorld evaluation script
│
├── get_trajectory/        # BFCL trajectory extraction for experience building
│
├── utils/                 # Shared utility modules
│
├── run.sh                # Main evaluation script
├── utils.py              # Core utilities
└── io_utils.py           # I/O utilities

```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key-here"
export CUDA_VISIBLE_DEVICES="0,1"
```

### Basic Usage

```bash
# Run BFCL evaluation
./run.sh
```

---

## 📚 Module Documentation

### 1. **bfcl_eval** - Core Evaluation Framework

Berkeley Function Calling Leaderboard evaluation system.

#### Key Components:

**Model Handlers** (`bfcl_eval/model_handler/`)
- `local_inference/qwen.py` - Qwen model handlers
  - `QwenHandler` - Base prompt mode
  - `MA_QwenEnhanceHandler` - Enhanced mode with retrieval
- `local_inference/qwen_fc.py` - Qwen function calling handlers
  - `QwenFCHandler` - Function calling mode

**Configuration** (`bfcl_eval/constants/model_config.py`)
```python
# Register models
"Qwen/Qwen3-8B-FC": ModelConfig(
    model_handler=QwenFCHandler,
    is_fc_model=True,
)
```


---

### 2. **Experience Management**


#### **A. BFCL Experience Building** (`get_trajectory/`)

Used for building experience knowledge base from BFCL evaluation results.

**Workflow:**

1. **Extract trajectories** from successful BFCL runs:
```bash
python get_trajectory/extract_trajectory.py \
    --input result/result_fc/Qwen_Qwen3-8B-FC/multi_turn/ \
    --output Experience/BFCL_experience.json
```

2. **Generate summaries** using GPT:
```bash
python get_trajectory/get_trajectory_summary.py \
    --input Experience/BFCL_experience.json \
    --output Experience/BFCL_experience_summary.json \
    --model "gpt-4o"
```

3. **Add embeddings** for retrieval:
```bash
python get_trajectory/add_embedding.py \
    --input Experience/BFCL_experience_summary.json \
    --output Experience/BFCL_experience_with_embedding.json \
    --model-path "all-MiniLM-L6-v2"
```

**Output Format (BFCL):**
```json
{
  "sample_id": "multi_turn_001",
  "question": [
    [{"role": "user", "content": "Search for files..."}],
    [{"role": "user", "content": "Now filter by date..."}]
  ],
  "trajectory": "Step 1: Call search_files()...\nStep 2: Call filter_by_date()...",
  "summary": "For file search tasks, always verify path existence...",
  "embedding": [0.1, 0.2, ...],
  "tool_calls": [...]
}
```

---

#### **B. AppWorld Experience Building** (`app_world/`)

Used for building experience knowledge base from AppWorld benchmark results.

**Workflow:**

1. **Run AppWorld evaluation** to generate trajectories:
```bash
cd app_world
python appworld_test.py \
    --dataset_name "appworld_test" \
    --experiment_name "qwen3_baseline"
```

2. **Generate summaries** using GPT:
```bash
python app_world/get_trajectory_summary.py \
    --input app_world/Experience/appworld_raw.json \
    --output app_world/Experience/appworld_summary.json \
    --model "gpt-4o"
```

3. **Add embeddings** for retrieval:
```bash
python app_world/add_embedding.py \
    --input app_world/Experience/appworld_summary.json \
    --output app_world/Experience/appworld_with_embedding.json \
    --model-path "/path/to/all-MiniLM-L6-v2"
```

---

### 3. **run.sh** - Main Evaluation Script

#### BFCL

Automated evaluation pipeline for multiple configurations.

**Script Workflow:**

```bash
#!/bin/bash

# 1. Set environment
export CUDA_VISIBLE_DEVICES="7,8"
export API_KEY="EMPTY"

# 2. Configure models and tests
MODELS=("Qwen/Qwen3-8B-FC")
TEST_CATEGORY="multi_turn_base_testing,multi_turn_miss_param_testing"
INFORMATION_DICT="./Experience/experience_with_embedding.json"

# 3. Run evaluations
for MODEL in "${MODELS[@]}"; do
    # Baseline
    bfcl generate --model "$MODEL" --result-dir result/result_fc
    bfcl evaluate --model "$MODEL" --result-dir result/result_fc
    
    # Enhanced with retrieval
    bfcl generate --model "$MODEL" \
        --if-enhanced \
        --enhanced-method "summary" \
        --information-dict "$INFORMATION_DICT" \
        --result-dir result/result_enhanced
    bfcl evaluate --model "$MODEL" --result-dir result/result_enhanced
done
```

**Customization:**

```bash
# Test single model
MODELS=("Qwen/Qwen3-8B-FC")

# Test multiple models
MODELS=("Qwen/Qwen3-8B-FC" "Qwen/Qwen3-32B-FC" "meta-llama/Llama-3.1-8B-Instruct-FC")

# Custom test categories
TEST_CATEGORY="multi_turn_base_testing"  # Single category
TEST_CATEGORY="multi_turn_base_testing,multi_turn_miss_param_testing"  # Multiple

# Change output directories
--result-dir result/my_experiment_results
--score-dir result/my_experiment_scores
```


#### **AppWorld Integration** (`app_world/`)

Evaluation on the AppWorld benchmark.

**Usage:**

```bash
cd app_world

# Run AppWorld evaluation
./run.sh

# Or run specific configuration
python memp.py \
    --dataset_name "appworld_test" \
    --experiment_name "qwen3_enhanced" \
    --information-dict "./Experience/appworld_experience.json"
```

**AppWorld Scripts:**
- `appworld_test.py` - Main evaluation script
- `memp.py` 
- `run.sh` 

---


## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
export OPENAI_API_KEY="sk-..."           # OpenAI API key
export AZURE_API_KEY="..."               # Azure API key (if using Azure)
export DASHSCOPE_API_KEY="..."          # Qwen API key (for API mode)

# Model Configuration
export LOCAL_MODEL_PATH="/path/to/model"  # Local model path
export BASE_URL="http://localhost:8000/v1/"  # vLLM server URL
export API_KEY="EMPTY"                    # For local vLLM

# Hardware Configuration
export CUDA_VISIBLE_DEVICES="0,1"        # GPU devices
export HTTP_PROXY="..."                  # Proxy (if needed)
export HTTPS_PROXY="..."                 # HTTPS proxy

# Experience Configuration
export INFORMATION_DICT="./Experience/experience_with_embedding.json"
```

### Test Categories

Available test categories in BFCL:
- `multi_turn_base_testing` - Basic multi-turn tool calling
- `multi_turn_miss_param_testing` - Missing parameter scenarios
- `multi_turn_miss_func_testing` - Missing function scenarios
- `multi_turn_long_context_testing` - Long context handling
- `single_turn_base_testing` - Single-turn tool calling
- `single_turn_relevance_testing` - Relevance detection

---

## 📊 Results and Outputs

### Directory Structure

```
result/
├── result_fc/              # Function calling results
│   └── Qwen_Qwen3-8B-FC/
│       └── multi_turn/
│           └── BFCL_v4_multi_turn_base_testing.json
├── score_fc/               # Evaluation scores
│   └── Qwen_Qwen3-8B-FC/
│       └── multi_turn/
│           └── BFCL_v4_multi_turn_base_testing_score.json
└── result_enhanced/        # Enhanced method results
```
