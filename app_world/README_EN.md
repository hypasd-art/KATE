# KATE App_World Project Documentation

## Core Files Overview

### 1. Experience Directory
Stores experience data including historical task execution trajectories and summaries.

**File List:**
- `experience.json` - Raw experience data containing task questions, execution trajectories, embedding vectors, and required apps
- `experience_summary.json` - Experience summaries with GPT-generated 100-word abstracts for each task
- `minimal_react_agent_train.json` - Complete dialogue history for training set tasks

### 2. appworld_test.py
**Main Testing Script** - Runs tasks in AppWorld environment using ReAct paradigm

**Core Features:**
- **MinimalReactAgent Class**: Implements ReAct-based agent
  - Supports retrieval enhancement (retrieves similar tasks from experience library)
  - Supports parallel decoding (generates multiple candidate codes and aggregates)
  - Generates code step-by-step, one chunk at a time
  
**Key Features:**
- Retrieval Enhancement (`--retrieval_enhance`): Uses semantic similarity to retrieve top-k similar tasks from experience library
- Parallel Decoding (`--parallel_decode`): Generates N candidate solutions simultaneously, deduplicates and aggregates
- Multi-process Execution: Runs multiple tasks in parallel using ProcessPoolExecutor

**Execution Flow:**
1. Load tasks and experience data
2. Create Agent instance for each task
3. Iterate: generate code → execute → get feedback
4. Until task completes or max interactions reached

### 3. appworld_trajectory.py
**Trajectory Generation Script** - Generates dialogue trajectories based on ground truth code

**Core Features:**
- Retrieves standard solution code from AppWorld's ground truth
- Uses LLM to convert complete code into step-by-step dialogue format
- Generates code + prints intermediate results for each step

**Purpose:**
- Generates high-quality example trajectories for training set tasks
- Stores trajectories in `minimal_react_agent_train.json`
- Serves as foundation for retrieval enhancement experience library

### 4. get_trajectory.py
**Experience Extraction Script** - Extracts structured experiences from trajectory files

**Core Features:**
- Reads `minimal_react_agent_train.json`
- Extracts for each task:
  - User question
  - Code execution trajectory (step-by-step)
  - Question embedding vectors (for retrieval)
  - Required API applications list
- Outputs to `experience.json`

**Data Structure:**
```json
{
  "question text": {
    "question": "...",
    "trajectory": "Code of Step 0:\n...\n---\nCode of Step 1:\n...",
    "embedding": [0.1, 0.2, ...],
    "required_apps": ["spotify", "supervisor"]
  }
}
```

### 5. get_trajectory_summary.py
**Experience Summarization Script** - Generates abstract summaries for each experience

**Core Features:**
- Uses GPT-4 to analyze trajectories for each task
- Generates 100-word summaries including:
  - Concrete causal analysis (how the problem was solved)
  - Transferable general rules (tool selection, parameter handling, validation strategies)
  - Reasoning patterns from concrete to abstract

**Concurrent Processing:**
- Uses ThreadPoolExecutor (32 threads) for parallel summary generation
- Progress bar display
- Outputs to `experience_summary.json`

### 6. memp.py
**MEMP Method Implementation** - Memory-Enhanced Multi-step Planning

**Differences from appworld_test.py:**
- Does not support parallel decoding (single-path generation)
- Same retrieval enhancement mechanism
- Simplified Agent implementation

**Use Case:**
- Baseline method for comparison
- Reference implementation for single-path reasoning

## Data Flow Diagram

```
1. Generate Trajectories
   appworld_trajectory.py
   ↓
   minimal_react_agent_train.json

2. Extract Experiences
   get_trajectory.py
   ↓
   experience.json

3. Generate Summaries
   get_trajectory_summary.py
   ↓
   experience_summary.json

4. Test & Evaluate
   appworld_test.py / memp.py
   (uses experience.json for retrieval enhancement)
   ↓
   experiments/outputs/{experiment_name}/
```

## Key Technologies

### Retrieval Enhancement
1. Encodes questions using `all-MiniLM-L6-v2` model
2. Calculates cosine similarity, filters top-3 tasks with score > 0.5
3. Inserts similar task trajectories into prompt as reference

### Parallel Decoding
1. Generates N candidate codes simultaneously
2. Normalizes and deduplicates (removes comments and whitespace)
3. If all candidates identical → use directly
4. Otherwise → calls LLM to aggregate and select optimal code

### Step-by-Step Generation
- Generates only a small code chunk each time
- Gets feedback after execution
- Continues generating next step based on feedback
- Avoids generating complete solution at once

## Usage Examples

```bash
# With retrieval enhancement + parallel decode 4
./run.sh --gpt4 -r -p 4 -d test_normal

# Equivalent Python command
python appworld_test.py \
  --dataset_name test_normal \
  --experiment_name "test_normal_gpt-4.1_retrieval_True_4" \
  --retrieval_enhance \
  --parallel_decode 4 \
  --restart

# Using MEMP method
./run.sh --memp -r -d test_normal
```

## Requirements

- Python 3.8+
- sentence-transformers
- openai
- appworld package
- Model file: `all-MiniLM-L6-v2`

## Summary Table

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| appworld_trajectory.py | Generate dialogue trajectories | Ground truth code | minimal_react_agent_train.json |
| get_trajectory.py | Extract structured experiences | minimal_react_agent_train.json | experience.json |
| get_trajectory_summary.py | Generate experience summaries | experience.json | experience_summary.json |
| appworld_test.py | Main test script (retrieval + parallel decode) | Tasks + experience.json | Experiment results |
| memp.py | MEMP baseline method | Tasks + experience.json | Experiment results |

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Training Phase                          │
├─────────────────────────────────────────────────┤
│ Ground Truth → Trajectory → Experience → Summary│
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          Testing Phase                           │
├─────────────────────────────────────────────────┤
│ New Task → Retrieve Similar → Agent → Solution  │
└─────────────────────────────────────────────────┘
```
