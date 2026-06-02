import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from utils import OpenAI
from utils_prompt import _format_prompt

openai.api_key = os.getenv("OPENAI_API_KEY", "")
openai.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

prompt_template_display = """
**Task Input:**

* **Question and Inference Trajectory:** {question_and_inference_trajectory}

将上述内容翻译成中文
"""


def extract_turn_trajectory(inputs, turn_index):
    inference_log = inputs["inference_log"]
    previous_dialogue = []
    current_error_dialogue = {}
    num = 0

    for index, item in enumerate(inference_log):
        if index % 2 == 0:
            assert type(item) == list
        else:
            assert type(item) == dict
            if num == turn_index:
                new_item = {}
                for key, value in item.items():
                    if key.startswith("step_"):
                        new_item[key] = [v for v in value if v["role"] != "handler_log"]
                    else:
                        new_item[key] = value
                current_error_dialogue = new_item
            elif num < turn_index:
                for key, value in item.items():
                    if key.startswith("step_"):
                        for value_item in value:
                            if value_item["role"] == "assistant":
                                previous_dialogue.append({"role": "assistant", "content": value_item["content"]})
                            elif value_item["role"] == "tool":
                                previous_dialogue.append({"role": "tool", "content": value_item["content"]})
                    else:
                        assert value[0]["content"].startswith(
                            inputs["prompt"]["question"][num][0]["content"]
                        )
                        previous_dialogue.append({
                            "role": "user",
                            "content": inputs["prompt"]["question"][num][0]["content"],
                        })
            num += 1

    return {"previous_dialogue": previous_dialogue, "current_error_dialogue": current_error_dialogue}


def get_each_turn_information(information_inputs):
    information = {}
    for inputs in information_inputs:
        error = inputs["error"]
        error_message = error["error_message"]
        try:
            assert error_message[-11:-2] == "for turn "
        except AssertionError:
            continue
        turn_index = int(error_message[-2])

        inference_trajectory = extract_turn_trajectory(inputs, turn_index)
        question = inputs["prompt"]["question"][turn_index][0]["content"]
        information[question] = {
            "id_new": f"{inputs['id']}_turn_{turn_index}",
            "id": inputs["id"],
            "turn": turn_index,
            "question": question,
            "model_response": inputs["model_result_decoded"][turn_index],
            "answer": inputs["possible_answer"][turn_index],
            "correctness": False,
            "inference_trajectory": inference_trajectory,
            "inference_log": inputs["inference_log"],
        }
    return information


def gpt_analysis(inputs, model=DEFAULT_MODEL):
    full_prompt = prompt_template_display.format(
        question_and_inference_trajectory=json.dumps(inputs["inference_trajectory"], indent=2),
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

    return {
        "model_tool-calling_trajectory": inputs["model_response"],
        "correct_answer": inputs["answer"],
        "translation": response.choices[0].message.content if response else "",
    }


def process_single(client, process_result, model=DEFAULT_MODEL):
    response = gpt_analysis(process_result, model=model)
    return {"question": process_result["question"], "response": response}


def get_analysis(file_path, output_path, model=DEFAULT_MODEL, max_workers=32):
    print(f"Processing {file_path}...")

    results = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if "accuracy" not in data:
                    results.append(data)

    results_new = get_each_turn_information(results)

    output = []
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in results_new.values():
            if not result["correctness"]:
                futures.append((result["id"], executor.submit(process_single, None, result, model)))

        for rid, future in tqdm(futures, total=len(futures), desc="Analyzing results"):
            res = future.result()
            if res is not None:
                output.append(res)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input_file", type=str, required=True, help="Path to the score result JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    get_analysis(args.input_file, args.output_file, model=args.model, max_workers=args.max_workers)
