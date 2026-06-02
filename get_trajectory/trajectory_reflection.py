import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from sentence_transformers import SentenceTransformer

from prompt import MULTI_TURN_FUNC_DOC_FILE_MAPPING_N, OMIT_STATE_INFO_CLASSES, involved_classes_state

openai.api_key = os.getenv("OPENAI_API_KEY", "")
openai.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "/home/yphao/Experience_Tool/test/gorilla/berkeley-function-call-leaderboard/all-MiniLM-L6-v2"
)

prompt_template_reflection = """For the user's question and the model's response trajectory, you need to determine whether the task has been completed.

If it has been completed and the model's response trajectory is correct, you should provide a summary.

If it has not been completed or the model's response trajectory is incorrect, you should provide a reflection.

**Task Input:**

* **available Tools:** {tools}
* **Task Config:** {task_config}
* **Question and Inference Trajectory:** {question_and_inference_trajectory}

Task: Produce a single, continuous explanation of **exactly 100 words**.

**Summary Guidelines:**

* **Start with concrete, current-turn analysis.** Diagnose the causal factors behind the (correct) resolution in the current turn: how the problem was interpreted, constraints extracted, tools selected (and parameters), evidence gathered/validated, intermediate checks, error handling, and the final verification that justified the answer.
* **Then generalize into transferable rules, but include concrete details.** Elevate those observations into reusable heuristics: reasoning patterns, tool-selection criteria, validation loops, uncertainty management, fallback strategies, and stopping/decision rules that apply across similar scenarios.
* **Use previous turns only as supporting context**.
* **Keep it concise and task-focused,** avoiding digressions or narrative filler.
* **Deliver one cohesive paragraph** that flows from concrete diagnosis to abstract guidance to principle extraction.

**Reflection Guidelines:**

* **Begin** with the direct cause of the error (this must be the first sentence).
* Then analyze the **underlying causes** beyond surface symptoms (for example: failing to check current state, omitting implicit actions, wrong call order, context loss across turns, parameter misuse, or overplanning).
* Finish with **detailed, targeted advice** that addresses the stated direct cause and underlying causes, and prescribes concrete mitigations for two causes.

The recommended output format is:
Summary Format: The summary is: ...
Reflection Format: The problem is: … when asking this question, The direct cause of this problem is: … The root cause is: … To avoid this problem, when encountering similar cases in the future, one should …

Formatting rules: plain text only, no headings or code blocks, no extra commentary. Ensure the response is exactly 100 words and focused on actionable, cause-aligned recommendations.
"""


def prepare_tools(path):
    json_contents = {}
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory")

    for filename in os.listdir(path):
        if not filename.endswith(".json"):
            continue
        if filename in OMIT_STATE_INFO_CLASSES:
            continue
        file_path = os.path.join(path, filename)
        try:
            json_tools = {}
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        json_tools[item["name"]] = item
            json_contents[MULTI_TURN_FUNC_DOC_FILE_MAPPING_N[filename]] = json_tools
        except json.JSONDecodeError:
            print(f"Warning: {filename} is not valid JSON, skipped")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return json_contents


def gpt_analysis(inputs, model=DEFAULT_MODEL):
    tools = inputs["apis"]
    task_config = inputs["initial_config"]
    inference_log = inputs["inference_log"]

    previous_dialogue = []
    for index, item in enumerate(inference_log):
        if index % 2 == 0:
            assert type(item) == list
        else:
            assert type(item) == dict
            for key, value in item.items():
                if key.startswith("step_"):
                    for value_item in value:
                        if value_item["role"] == "assistant":
                            previous_dialogue.append({"role": "assistant", "content": value_item["content"]})
                        elif value_item["role"] == "tool":
                            previous_dialogue.append({"role": "tool", "content": value_item["content"]})
                else:
                    previous_dialogue.append({"role": "user", "content": value[0]["content"]})

    question_and_inference_trajectory = {
        "question": inputs["question"],
        "model_tool-calling_trajectory": previous_dialogue,
    }

    full_prompt = prompt_template_reflection.format(
        task_config=json.dumps(task_config),
        tools=json.dumps(tools),
        question_and_inference_trajectory=json.dumps(question_and_inference_trajectory, indent=2),
    )
    messages = [{"role": "user", "content": full_prompt}]
    response = None
    for attempt in range(3):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            break
        except Exception as e:
            print(e)

    action = [
        value_item["content"]
        for index, item in enumerate(inference_log)
        if index % 2 == 1
        for key, value in item.items()
        if key.startswith("step_")
        for value_item in value
        if value_item["role"] == "assistant"
    ]

    return {
        "model_tool-calling_trajectory": previous_dialogue,
        "reflection": response.choices[0].message.content if response else "",
        "action": action,
    }


def process_single(process_result, model=DEFAULT_MODEL):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    try:
        response = gpt_analysis(process_result, model=model)
        return {
            "question": process_result["question"],
            "response": response,
            "all_tools": process_result["apis"],
            "tools": process_result["tools"],
            "involved_classes": process_result["involved_classes"],
            "initial_config": process_result["initial_config"],
            "embedding": embedding_model.encode([process_result["question"]])[0].tolist(),
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_item(tep_item, model=DEFAULT_MODEL):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Conclude the intent of the user's question in one sentence."},
                {"role": "user", "content": "Question: " + tep_item["question"]},
            ],
            temperature=0,
        )
        tep_item["intent"] = response.choices[0].message.content
        tep_item["embedding"] = embedding_model.encode([tep_item["question"]])[0].tolist()
        tep_item["intent_embedding"] = embedding_model.encode([tep_item["intent"]])[0].tolist()
        return tep_item
    except Exception as e:
        print(f"Error processing {tep_item.get('question')}: {e}")
        return None


def run_multithread(output, model=DEFAULT_MODEL, max_workers=32):
    t_n = {k: {} for k in involved_classes_state.keys()}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item, model): item for item in output}
        for future in as_completed(futures):
            tep_item = future.result()
            if tep_item:
                t_n[tep_item["involved_classes"][0]][tep_item["question"]] = tep_item
    return t_n


def get_analysis(file_path, output_path, model=DEFAULT_MODEL, max_workers=32):
    with open(file_path, "r") as f:
        output = json.load(f)

    question_groups = defaultdict(list)
    for item in output:
        question_groups[item.get("question")].append(item)
    duplicates = {q: items for q, items in question_groups.items() if len(items) > 1}
    for q, items in duplicates.items():
        print(f"Duplicate question: {q} ({len(items)} entries)")

    t_n = run_multithread(output, model=model, max_workers=max_workers)

    with open(output_path, "w") as f:
        json.dump(t_n, f, indent=2, ensure_ascii=False)

    return t_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input_file", type=str, required=True, help="Path to the model result JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    get_analysis(args.input_file, args.output_file, model=args.model, max_workers=args.max_workers)
