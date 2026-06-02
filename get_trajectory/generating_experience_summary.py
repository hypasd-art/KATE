import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from sentence_transformers import SentenceTransformer

openai.api_key = os.getenv("OPENAI_API_KEY", "")
openai.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "/home/yphao/Experience_Tool/test/berkeley-function-call-leaderboard/all-MiniLM-L6-v2"
)

prompt_template_summary = """The following is the test result of an agent's tool-usage task, along with a correct model response.

**Input:**

* Previous turns before the current turn: {previous_turns}
* Current turn: {current_turn}

**Task:**
Write a **100-word summary** identifying the key factors needed to solve the user's question correctly. The goal is to ensure the model consistently provides accurate responses and tool calls in future tasks.

**Guidelines (Concrete → Abstract):**

* **Start with concrete, current-turn analysis.** Diagnose the causal factors behind the (correct) resolution in the current turn: how the problem was interpreted, constraints extracted, tools selected (and parameters), evidence gathered/validated, intermediate checks, error handling, and the final verification that justified the answer.
* **Then generalize into transferable rules, but include concrete details.** Elevate those observations into reusable heuristics: reasoning patterns, tool-selection criteria, validation loops, uncertainty management, fallback strategies, and stopping/decision rules that apply across similar scenarios.
* **Use previous turns only as supporting context**.
* **Keep it concise and task-focused,** avoiding digressions or narrative filler.
* **Deliver one cohesive paragraph** that flows from concrete diagnosis to abstract guidance to principle extraction.

"""


def merge_training_files(training_file, answer_file, output_path):
    data_dict = {}
    process_data_dict = {}

    with open(training_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data_dict[item["id"]] = item

    with open(answer_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item["id"] in data_dict:
                    data_dict[item["id"]].update(item)
                    for index, (question, answer) in enumerate(
                        zip(data_dict[item["id"]]["question"], data_dict[item["id"]]["ground_truth"])
                    ):
                        process_data_dict[question[0]["content"]] = {
                            "id": item["id"],
                            "turn": index,
                            "question": question[0]["content"],
                            "answer": answer,
                            "summary": "",
                            "all_question": data_dict[item["id"]]["question"],
                            "all_answer": data_dict[item["id"]]["ground_truth"],
                            "involved_classes": data_dict[item["id"]]["involved_classes"],
                        }
                        assert len(question) == 1
                else:
                    raise Exception(f"ID {item['id']} not found in the first file")

    with open(output_path, "w") as f:
        json.dump(process_data_dict, f, indent=2)

    return process_data_dict


def gpt_analysis(inputs, model=DEFAULT_MODEL):
    previous_turns = []
    current_turn = {"user_question": inputs["question"], "ground_truth": inputs["answer"]}
    for i in range(inputs["turn"]):
        previous_turns.append({
            "user_question": inputs["all_question"][i],
            "ground_truth": inputs["all_answer"][i],
        })
    full_prompt = prompt_template_summary.format(
        previous_turns=previous_turns,
        current_turn=current_turn,
    )
    messages = [{"role": "user", "content": full_prompt}]
    response = ""
    for attempt in range(3):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
            response = resp.choices[0].message.content
            break
        except Exception as e:
            print(e)
    return response


def process_result(result, model=DEFAULT_MODEL):
    try:
        response = gpt_analysis(result, model=model)
        result["summary"] = response
        return result
    except Exception as e:
        print(f"Error processing result: {e}")
        return None


def get_analysis(file_path, output_path, model=DEFAULT_MODEL, max_workers=32):
    print(f"Processing {file_path}...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    with open(file_path, "r") as f:
        information_dict = json.load(f)

    output = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_result = {
            executor.submit(process_result, result, model): result
            for result in information_dict.values()
        }
        for future in tqdm(as_completed(future_to_result), total=len(information_dict), desc="Analyzing results"):
            res = future.result()
            if res is not None:
                output.append(res)

    for item in output:
        question = item["question"]
        assert question in information_dict
        information_dict[question]["summary"] = item["summary"]

    for question, item in information_dict.items():
        embedding = embedding_model.encode([question])[0]
        item["embedding"] = embedding.tolist()

    with open(output_path, "w") as f:
        json.dump(information_dict, f, indent=2)

    return information_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--training_file", type=str, default="../bfcl_eval/data/BFCL_v4_multi_turn_base_training.json")
    parser.add_argument("--answer_file", type=str, default="../bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base_training.json")
    parser.add_argument("--merged_output", type=str, default="../Experience/BFCL_v4_multi_turn_base_training_summary_dict.json")
    parser.add_argument("--output_path", type=str, default="../Experience/BFCL_v4_multi_turn_base_training_summary_with_embedding.json")
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    merge_training_files(args.training_file, args.answer_file, args.merged_output)
    get_analysis(args.merged_output, args.output_path, model=args.model, max_workers=args.max_workers)
