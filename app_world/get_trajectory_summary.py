import json
from tqdm import tqdm
import os
import re
import argparse
import json
import ast 
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from sentence_transformers import SentenceTransformer, util
import openai
from appworld import AppWorld, load_task_ids
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
MODEL = os.getenv("MODEL", "gpt-4o")

model = SentenceTransformer('/home/yphao/Experience_Tool/test/berkeley-function-call-leaderboard/all-MiniLM-L6-v2')

information_file = "../Experience/BFCL_v4_multi_turn_base_training_summary_dict.json"


prompt_template_summary = """The following is the test result of an agent's tool-usage task, along with a correct model response.

**Input:**

* Question: {question}
* Answer: {answer}

**Task:**
Write a **100-word summary** identifying the key factors needed to solve the user's question correctly. The goal is to ensure the model consistently provides accurate responses and tool calls in future tasks.

**Guidelines (Concrete → Abstract):**

* **Start with concrete, current-turn analysis.** Diagnose the causal factors behind the (correct) resolution in the current turn: how the problem was interpreted, constraints extracted, tools selected (and parameters), evidence gathered/validated, intermediate checks, error handling, and the final verification that justified the answer (e.g., how it extracted constraints (e.g., directory `temp`, file `final_report.pdf`, keyword `budget analysis`), selected the appropriate tool (`grep`), configured the correct parameters, and validated that the output aligned with the user’s request.).
* **Then generalize into transferable rules, but include concrete details.** Elevate those observations into reusable heuristics: reasoning patterns, tool-selection criteria, validation loops, uncertainty management, fallback strategies, and stopping/decision rules that apply across similar scenarios. For example: when handling **file search tasks**, always consider `grep` first and verify that both the file and path exist before execution; for **parameter handling**, map context-derived keywords and file names directly to tool inputs to avoid omissions; for **validation**, confirm correctness by checking whether search results explicitly match the requested keyword.
* **Use previous turns only as supporting context**.
* **Keep it concise and task-focused,** avoiding digressions or narrative filler, and ensure the summary directly improves future reasoning and tool choices.
* **Deliver one cohesive paragraph** that flows from concrete diagnosis to abstract guidance to principle extraction, forming an actionable loop.

"""


def gpt_analysis(client, inputs):
    full_prompt = prompt_template_summary.format(question=inputs["question"], answer=inputs["trajectory"])
    response = ""
    attempt = 0
    messages = [{"role": "user", "content": full_prompt}]
    while attempt < 3:
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7
            )
            response = response.choices[0].message.content
        except Exception as e:
            print(e)
        attempt += 1
    print(response)
    return response

def process_result(client, result):
    """单个任务处理函数"""
    try:
        response = gpt_analysis(client, result)
        result["summary"] = response
        return result
    except Exception as e:
        print(f"Error processing result: {e}")
        return None

def get_analysis(file_path, client, output_path):
    print(f"Processing {file_path}...")
        
    information_dict = []
    with open(file_path, "r") as f:
        information_dict = json.load(f)
        
    output = []


    # 使用线程池
    with ThreadPoolExecutor(max_workers=32) as executor:
        # 提交所有任务
        future_to_result = {
            executor.submit(process_result, client, result): result
            for key, result in information_dict.items()
        }

        # 使用 tqdm 进度条跟踪完成情况
        for future in tqdm(as_completed(future_to_result), total=len(information_dict), desc="Analyzing results"):
            res = future.result()
            if res is not None:
                output.append(res)
    for item in output:
        question = item["question"]
        assert question in information_dict
        information_dict[question]["summary"] = item["summary"]
    output = information_dict
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

get_analysis("./Experience/experience.json", "", "./Experience/experience_summary.json")

