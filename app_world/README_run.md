# run.sh Documentation

## Quick Start

```bash
# Default configuration (Qwen3-32B, test_normal)
./run.sh

# GPT-4 with retrieval augmentation and 4 parallel workers
./run.sh --gpt4 -r -p 4

# Run both datasets
./run.sh -d both

# Use Qwen3-8B
./run.sh --qwen8b
```

## Environment Variables

Before running, export the appropriate base URL for your model server:

```bash
export QWEN32B_BASE_URL="http://your-server:port/v1"   # for --qwen32b (default)
export QWEN8B_BASE_URL="http://your-server:port/v1"    # for --qwen8b
export OPENAI_BASE_URL="https://api.openai.com/v1"     # for --gpt4 (optional, this is the default)
export OPENAI_API_KEY="sk-..."                         # for --gpt4
```

## Options

### Dataset
- `-d, --dataset DATASET` — dataset to evaluate
  - `test_normal` (default)
  - `test_challenge`
  - `both` — run both datasets sequentially

### Model Presets
- `--gpt4` — GPT-4.1-2025-04-14
- `--qwen32b` — Qwen3-32B (default)
- `--qwen8b` — Qwen3-8B

### Run Configuration
- `-p, --parallel NUM` — number of parallel decode workers (default: 1)
- `-r, --retrieval` — enable retrieval-augmented mode
- `--restart` — restart the experiment from scratch

### Test Script
- Default: `appworld_test.py`
- `--memp` — use `memp.py`
- `--test2` — use `appworld_test_2.py`

## Examples

### Basic Usage
```bash
# Default run
./run.sh

# Specific dataset
./run.sh -d test_challenge

# Enable retrieval
./run.sh -r

# Set parallel workers
./run.sh -p 4
```

### Combined Options
```bash
# GPT-4 + retrieval + 4 workers + both datasets
./run.sh --gpt4 -r -p 4 -d both

# Qwen3-8B + retrieval + restart
./run.sh --qwen8b -r --restart

# memp.py + retrieval
./run.sh --memp -r

# test_challenge + test2 script + 4 workers
./run.sh --test2 -d test_challenge -p 4
```

## Experiment Naming

The script auto-generates experiment names:

- **appworld_test.py**: `{dataset}_{model}_retrieval_{True/False}_{parallel}`
  - e.g. `test_normal_Qwen3-32B_retrieval_True_4`

- **memp.py**: `memp_{dataset}_{model}`
  - e.g. `memp_test_normal_Qwen3-32B`

- **appworld_test_2.py**: `m_{dataset}_{model}_retrieval_{True/False}_{parallel}`
  - e.g. `m_test_challenge_gpt-4.1-2025-04-14_retrieval_True_4`

## Workflow

Each experiment runs the following steps:

1. Validate that `BASE_URL` is set
2. Export environment variables (`BASE_URL`, `KEY`, `MODEL`)
3. Generate the experiment name
4. Run the Python test script
5. Evaluate results with `appworld evaluate`

## Common Combinations

| Scenario | Command |
|----------|---------|
| Quick test | `./run.sh` |
| Full evaluation | `./run.sh -d both -r -p 4` |
| GPT-4 baseline | `./run.sh --gpt4 -d both` |
| Model comparison | `./run.sh --qwen32b -d both && ./run.sh --qwen8b -d both` |
| MEMP method | `./run.sh --memp -r -d both` |

## Notes

1. Ensure the required Python script exists before running
2. Ensure the `appworld` CLI tool is installed
3. The model API endpoint must be reachable
4. Results are automatically saved to the corresponding output directory
