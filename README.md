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
│
├── app_world/             # AppWorld benchmark integration
│   ├── data/             # AppWorld test data
│   ├── Experience/       # Experience storage for AppWorld
│   └── run.sh           # AppWorld evaluation script
│
├── Experience/            # Knowledge base for experience retrieval
│   ├── *.json           # Experience trajectories and summaries
│   └── *_with_embedding.json  # Experiences with embeddings for retrieval
│
├── error_analysis/        # Error analysis and visualization tools
│   ├── analysis_human.py  # GPT-based error analysis
│   ├── pic.py            # Visualization generation
│   ├── utils.py          # LMM utilities
│   └── io_utils.py       # File I/O utilities
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

# Or run specific tests
bfcl generate --model "Qwen/Qwen3-8B-FC" --test-category "multi_turn_base_testing"
bfcl evaluate --model "Qwen/Qwen3-8B-FC" --test-category "multi_turn_base_testing"
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
  - `Qwen2_5_FCHandler` - Qwen2.5 specific handler

**Configuration** (`bfcl_eval/constants/model_config.py`)
```python
# Register models
"Qwen/Qwen3-8B-FC": ModelConfig(
    model_handler=QwenFCHandler,
    is_fc_model=True,
)
```

**Usage:**
```bash
# Generate predictions
bfcl generate \
    --model "Qwen/Qwen3-8B-FC" \
    --test-category "multi_turn_base_testing" \
    --result-dir result/result_fc \
    --temperature 0

# Evaluate results
bfcl evaluate \
    --model "Qwen/Qwen3-8B-FC" \
    --test-category "multi_turn_base_testing" \
    --result-dir result/result_fc \
    --score-dir result/score_fc
```

**Enhanced Mode with Experience Retrieval:**
```bash
bfcl generate \
    --model "Qwen/Qwen3-8B" \
    --if-enhanced \
    --enhanced-method "summary" \
    --trajectory-retrieval \
    --information-dict "./Experience/experience_with_embedding.json" \
    --MA-prompt "default" \
    --temperature 0
```

#### Enhancement Methods:

1. **Summary Retrieval** (`--enhanced-method "summary"`)
   - Retrieves summarized experience knowledge
   - Fast and efficient

2. **Trajectory Retrieval** (`--trajectory-retrieval`)
   - Retrieves full execution trajectories
   - More detailed context

3. **Intent-based Retrieval** (`--enhanced-method "intent"`)
   - Extracts user intent and retrieves relevant patterns
   - Best for complex tasks

4. **Multi-Agent Mode** (`--MA-prompt "default"`)
   - Parallel sampling with multiple agents
   - Fusion methods: majority voting, critic-based

---

### 2. **Experience Management**

#### Directory: `Experience/`

**Files:**
- `BFCL_v4_*.json` - BFCL experience data
- `appworld_*.json` - AppWorld experience data
- `*_summary.json` - Summarized experiences
- `*_with_embedding.json` - Experiences with embeddings for retrieval

---

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

**Output Format (AppWorld):**
```json
{
  "task_id": "appworld_001",
  "question": "Create a new contact with name John...",
  "trajectory": "1. Call create_contact(name='John')...",
  "summary": "For contact creation, always validate required fields...",
  "embedding": [0.1, 0.2, ...],
  "api_calls": [...]
}
```

---

#### **Key Differences:**

| Aspect | BFCL (`get_trajectory/`) | AppWorld (`app_world/`) |
|--------|--------------------------|-------------------------|
| **Data Source** | BFCL evaluation results | AppWorld benchmark results |
| **Input Format** | Multi-turn conversations | Task-based queries |
| **Scripts Location** | `/get_trajectory/` | `/app_world/` |
| **Experience File** | `BFCL_v4_*.json` | `appworld_*.json` |
| **Tool Calls** | Function calling format | API call format |
| **Use Case** | General tool-use tasks | App-specific workflows |

**Experience Format:**
```json
{
  "question": "User query",
  "trajectory": "Step-by-step solution",
  "summary": "Key insights and patterns",
  "embedding": [0.1, 0.2, ...],
  "intent": "Extracted task intent"
}
```

---



---

### 4. **run.sh** - Main Evaluation Script

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

---

### 5. **AppWorld Integration** (`app_world/`)

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
- `appworld_test_2.py` - Alternative implementation
- `memp.py` - Multi-agent evaluation
- `run.sh` - Automated pipeline

---

## 🎯 Common Use Cases

### Case 1: Evaluate a New Model

```bash
# 1. Add model to model_config.py
# bfcl_eval/constants/model_config.py

# 2. Run evaluation
bfcl generate --model "YourOrg/YourModel-FC" \
    --test-category "multi_turn_base_testing" \
    --result-dir result/your_model

bfcl evaluate --model "YourOrg/YourModel-FC" \
    --result-dir result/your_model \
    --score-dir result/your_model_scores
```

### Case 2A: Build BFCL Experience Knowledge Base

```bash
# 1. Extract successful trajectories from BFCL evaluation results
python get_trajectory/extract_trajectory.py \
    --input result/result_fc/Qwen_Qwen3-8B-FC/multi_turn/ \
    --output Experience/BFCL_experience.json

# 2. Generate summaries using GPT
python get_trajectory/get_trajectory_summary.py \
    --input Experience/BFCL_experience.json \
    --output Experience/BFCL_experience_summary.json \
    --model "gpt-4o"

# 3. Add embeddings for retrieval
python get_trajectory/add_embedding.py \
    --input Experience/BFCL_experience_summary.json \
    --output Experience/BFCL_experience_with_embedding.json \
    --model-path "all-MiniLM-L6-v2"

# 4. Use in enhanced evaluation
bfcl generate \
    --model "Qwen/Qwen3-8B" \
    --if-enhanced \
    --information-dict "Experience/BFCL_experience_with_embedding.json"
```

### Case 2B: Build AppWorld Experience Knowledge Base

```bash
# 1. Run AppWorld evaluation to collect trajectories
cd app_world
python appworld_test.py \
    --dataset_name "appworld_test" \
    --experiment_name "baseline"

# 2. Generate summaries using GPT
python app_world/get_trajectory_summary.py \
    --input app_world/Experience/appworld_raw.json \
    --output app_world/Experience/appworld_summary.json

# 3. Add embeddings
python app_world/add_embedding.py \
    --input app_world/Experience/appworld_summary.json \
    --output app_world/Experience/appworld_with_embedding.json

# 4. Use in AppWorld evaluation
python appworld_test.py \
    --information-dict "app_world/Experience/appworld_with_embedding.json"
```

### Case 3: Enhanced Evaluation with Retrieval

```bash
bfcl generate \
    --model "Qwen/Qwen3-8B" \
    --if-enhanced \
    --enhanced-method "summary" \
    --trajectory-retrieval \
    --information-dict "./Experience/experience_with_embedding.json" \
    --result-dir result/enhanced_results
```

### Case 4: Multi-Agent Evaluation

```bash
bfcl generate \
    --model "Qwen/Qwen3-8B" \
    --if-enhanced \
    --MA-prompt "default" \
    --MA-sample-num 4 \
    --MA-fusion-method "critic" \
    --result-dir result/multi_agent
```

### Case 5: Error Analysis

```bash
# 1. Analyze errors
python error_analysis/analysis_human.py \
    --file_path result/score_fc/model_scores.json \
    --output_path error_analysis/analysis.json

# 2. Generate visualizations
python error_analysis/pic.py

# Check output: error_analysis/pic/*.png
```

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

### Result Format

**Prediction File** (`result_*/model_name/test.json`):
```json
{
  "id": "sample_001",
  "model_response": "[{\"name\": \"search\", \"arguments\": {...}}]",
  "model_response_raw": "...",
  "inference_log": [...],
  "latency": 1.23,
  "inference_tokens": 150
}
```

**Score File** (`score_*/model_name/test_score.json`):
```json
{
  "accuracy": 0.85,
  "sample_index": "001",
  "error": {
    "error_type": "format_error",
    "error_message": "Invalid JSON format"
  },
  "analysis": ["Error analysis", ["Intent Error"]]
}
```

---

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
```bash
# Error: OPENAI_API_KEY not found
# Solution:
export OPENAI_API_KEY="your-key"
```

2. **CUDA Out of Memory**
```bash
# Reduce batch size or use smaller models
export CUDA_VISIBLE_DEVICES="0"  # Use single GPU
```

3. **Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

4. **vLLM Server Not Found**
```bash
# Start vLLM server first
vllm serve /path/to/model --port 8000
```

5. **Experience File Not Found**
```bash
# Check file path
ls -la Experience/experience_with_embedding.json

# Or use absolute path
--information-dict "/full/path/to/experience.json"
```

---

## 📖 Advanced Topics

### Custom Model Handler

Create a new handler in `bfcl_eval/model_handler/local_inference/`:

```python
from .base_oss_handler import BaseOSSHandler

class MyModelHandler(BaseOSSHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        # Custom initialization
    
    def decode(self, messages, **kwargs):
        # Implement model-specific decoding
        pass
```

Register in `model_config.py`:
```python
"MyOrg/MyModel": ModelConfig(
    model_handler=MyModelHandler,
    is_fc_model=True,
)
```

### Custom Retrieval Method

Implement in `MA_QwenEnhanceHandler`:

```python
def custom_retrieval(self, query, k=3):
    # Your retrieval logic
    embeddings = self.model.encode(query)
    results = search_knowledge_base(embeddings, k)
    return results
```

---

## 📝 Citation

If you use KATE in your research, please cite:

```bibtex
@article{kate2024,
  title={KATE: Knowledge-Augmented Tool-use Enhancement},
  author={Your Team},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📄 Documentation Files

- `README_qwen_handlers.md` - Detailed Qwen handler documentation
- `CODE_ANALYSIS.md` - Error analysis code review (error_analysis/)
- Individual module READMEs in subdirectories

---

## 🔗 Related Resources

- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [AppWorld Benchmark](https://appworld.dev/)
- [Qwen Documentation](https://qwen.readthedocs.io/)

---

## 📞 Support

For issues and questions:
1. Check existing documentation in module subdirectories
2. Review error logs in `result/` directories
3. Consult model-specific handler documentation
4. Check environment variable configuration

---

**Last Updated:** 2026-06-02  
**Version:** 1.0  
**Maintainers:** KATE Team
