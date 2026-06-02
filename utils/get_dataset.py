# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from transformers import AutoTokenizer
import re
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
import copy


prompt_tool_call = """
You need to call the tool at this turn, the ground truth tool call is {t_ground}. Your output should be consistent with the ground truth tool call. No other parameters is needed in your output.
you are not allowed to show that you know the correct tool call in your response and reasoning process, but your output should be consistent with the ground truth tool call, including the tool name and parameters should be consistent strictly..
"""

prompt_response = """
You need to respond to the user at this turn, the ground truth response is {r_ground}. Your output should be consistent with the ground truth response.
you are not allowed to show that you know the correct response in your response and reasoning process, but your output should be consistent with the ground truth response, including the tool name and parameters should be consistent strictly.. 
"""

decide_tool_calling_prompt = """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- Candidate tool calls: {candidate_plans}  

**Return Format**  
   Return a JSON object with the following structure:

```json
{{
  "optimal_plan": "<Explain The optimal plan and tool calls to execute next (You don't need to explain why you choose this approach, but rather explain why you are executing this tool_call.)>",
  "optimal_tool_call": {{
    "name": "<tool name>",
    "parameters": {{}}
  }}
}}
```
Only one tool call is allowed in the optimal_tool_call.
If no tool call is needed, set "optimal_tool_call": {{"name": "response_to_user", "parameters": {{"content": "The response to the user"}}}}.
"""



decide_tool_calling_prompt_ground = """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- Candidate tool calls: {candidate_plans}  

**Return Format**  
   Return a JSON object with the following structure:

```json
{{
  "optimal_plan": "<Explain The optimal plan and tool calls to execute next (You don't need to explain why you choose this approach, but rather explain why you are executing this tool_call.)>",
  "optimal_tool_call": {{
    "name": "<tool name>",
    "parameters": {{}}
  }}
}}
Only one tool call is allowed in the optimal_tool_call.
If no tool call is needed, set "optimal_tool_call": {{"name": "response_to_user", "parameters": {{"content": "The response to the user"}}}}.

Attention the ground truth tool call is {ground_t}, if the ground truth tool call not in the candidate plans, you should generate the correct tool call or response by yourself.
Your optimal tool call should be consistent with the {ground_t}, including the tool name and parameters should be consistent strictly.
And you are not allowed to generate any content that reflect you know the ground truth tool call.
"""

MULTI_TURN_FUNC_DOC_PATH = "./bfcl_eval/data/multi_turn_func_doc"
MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}
tokenizer = AutoTokenizer.from_pretrained("/netcache/huggingface/Qwen2.5-1.5B-Instruct")
from sentence_transformers import SentenceTransformer, util
import numpy as np
model = SentenceTransformer('/home/yphao/Experience_Tool/test/gorilla/berkeley-function-call-leaderboard/all-MiniLM-L6-v2')

def normalize_mixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中的 list/dict/object 列统一转换为字符串，避免 parquet 保存报错。
    """
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return df

def load_file(file_path, sort_by_id=False, allow_concatenated_json=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            try:
                content = json.loads(line)
                result.append(content)
            except Exception as e:
                if not allow_concatenated_json:
                    raise e

    if sort_by_id:
        result.sort(key=lambda x: x["id"])
    return result



def top_k_similar_questions_reflection_and_summary(analysis_result, target_content, k=5, p = 0.6, involved_classes=None, skip_first_example=False):
    target_embeddings = model.encode([target_content])[0]
    information_results = []
    for involved_class in involved_classes:
        # print("involved_class:", involved_class)
        scores = []
        for question, item in analysis_result.items():
            if involved_class in item["involved_classes"]: # []
                question_embeddings = np.array(item["embedding"], dtype=np.float32) 
                score = util.cos_sim(target_embeddings, question_embeddings).item()
                if score > p:
                    scores.append((score, question))
        # 按分数降序排序，选前k个
        scores.sort(reverse=True)
        if skip_first_example:

            scores = scores[1:]
        information_results.extend(scores)

    information_results = sorted(information_results, key=lambda x:x[0], reverse=True)
    seen = set()
    results = []
    for score, question in information_results:
        if question not in seen:
            results.append((score, question))
            seen.add(question)
        if len(results) >= k:
            break
    results.sort(key=lambda x:x[0], reverse=True)

    information_results = [q for _, q in results] 
    return information_results


def to_tool_call_format(s: str) -> str:
    """
    将形如 find(path='.',name='test_document.txt')
    的字符串转换为 <tool_call> JSON 格式
    """
    match = re.match(r"(\w+)\((.*)\)", s.strip())
    if not match:
        raise ValueError("输入格式不正确，应类似于: func(a='x', b='y')")
    
    func_name, args_str = match.groups()
    
    args = {}
    for k, v in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str):
        args[k] = v

    tool_call = {
        "name": func_name,
        "arguments": args
    }

    json_str = json.dumps(tool_call, ensure_ascii=False)
    
    formatted = f"<tool_call>\n{json_str}\n</tool_call>" # \\
    return formatted
def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception as e:
                pass
        return result
from openai import OpenAI

MODEL_NAME = os.getenv("MODEL_NAME", "") 
API_KEY = os.getenv("API_KEY", "")  
base_url= os.getenv("BASE_URL", "") 
client = OpenAI(api_key=API_KEY, base_url=base_url)
def get_llm_response(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt, 
        temperature=0.7,
        max_tokens=8192
    )
    return response.choices[0].message.content
def get_llm_response_n(prompt, n_sample=1):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt, 
        temperature=0.7,
        max_tokens=8192,
        n=n_sample
    )
    return [choice.message.content for choice in response.choices]

def process_single_row(row, row_index):
    """

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    import numpy as np
    function = []
    involved_classes = row.get("involved_classes")
    for func_collection in involved_classes:
        # func_doc is a list of dict
        func_doc = load_file(
            MULTI_TURN_FUNC_DOC_PATH  + "/" + MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
        )
        function.extend(func_doc)
    # Handle Miss Func category; we need to remove the holdout function doc
    if "missed_function" in row and not pd.isna(row["missed_function"]):
        new_missed = {}
        for turn_index, missed_func_names in row["missed_function"].items():
            row["missed_function"][turn_index] = []
            for missed_func_name in missed_func_names:
                for i, func_doc in enumerate(function):
                    if func_doc["name"] == missed_func_name:
                        # Add the missed function doc to the missed_function list
                        row["missed_function"][turn_index].append(func_doc) # 
                        # Remove it from the function list
                        function.pop(i)
                        break
            #  = new_missed
    formatted_prompt = ""
    formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for tool in function:
        tool["description"] += " Note that the provided function is in Python 3 syntax."
        formatted_prompt += f"\n{json.dumps(tool)}"
    formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'

    question = row.get("question", "")
    processed_question = []
    if "missed_function" in row and not pd.isna(row["missed_function"]):
        for idx, item in enumerate(question):
            if str(idx) in row["missed_function"]:
                processed_question.append([{"role": "user", "content": json.dumps(row["missed_function"][str(idx)]) + "\nI have updated some more functions you can choose from. What about now?"}])
                assert len(item) == 0
            else:
                processed_question.append(item)
    else:
        processed_question = question

    try:
        # Build prompt structure
        prompt = [{"role": "system", "content": formatted_prompt}] 
        llm_tool_calls = []
        

        # Extract ground truth from reward_model or fallback to golden_answers
        ground_truth = row.get("golden_answers", [])
        initial_config = row.get("initial_config")
        involved_classes = row.get("involved_classes")
        test_entry_id = row.get("id")
        trajectory_id = ""
        for idx, q in enumerate(processed_question):
            prompt.append({"role": "user", "content": q[0]["content"]})
            if len(ground_truth[idx]) == 0:
                prompt_n = copy.deepcopy(prompt)
                # assert "miss_func" in row["id"] or "miss_param" in row["id"], "Ground truth is empty but not a miss_func or miss_param case."
                if "miss_func" in row["id"]:
                    content_response = "I am sorry, but I cannot provide a valid tool call as the required function is not available."
                elif "miss_param" in row["id"]:
                    content_response = "I am sorry, but I cannot provide a valid tool call as the required parameters are missing."
                attempt = 0
                prompt.append({"role": "assistant", "content": content_response})

            for item in ground_truth[idx]:
                tool_call_ground = to_tool_call_format(item)
                
                prompt.append({"role": "assistant", "content": tool_call_ground})
                parsed_action = [item]
                execution_results, involved_instances = execute_multi_turn_func_call(
                        parsed_action,
                        initial_config,
                        involved_classes,
                        "model",
                        test_entry_id,
                        long_context=True if "long_context" in test_entry_id else False,
                    )
                prompt.append({"role": "user", "content": "<tool_response>" + execution_results[0] + "</tool_response>"}) # >
                
            content_response = get_llm_response(prompt + [{"role": "user", "content": "Based on the above conversation, give the response by summarizing the information and do not call tools."}]) # .split("</think>")[-1].strip()
            content_response = content_response.split("</think>")[-1].strip()
            prompt.append({"role": "assistant", "content": content_response})
    except:
        print(json.dumps(prompt, indent=2))
        raise Exception
    return {"messages": prompt, "involved_classes": involved_classes}, llm_tool_calls


def main():


    import os
    import json
    import random
    import pandas as pd
    import numpy as np

    data_path = "./bfcl_eval/data"
    results = []
    selected_data = []

    data_types = [
        "BFCL_v4_multi_turn_base_training.json",
        "BFCL_v4_multi_turn_miss_param_training.json",
        "BFCL_v4_multi_turn_long_context_training.json",
        "BFCL_v4_multi_turn_miss_func_training.json"
    ]


    # 读取所有数据类型
    dataset = []
    type_datasets = {}
    for data_type in data_types:
        type_results = []

        # 读取主文件
        with open(os.path.join(data_path, data_type), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    type_results.append(data)

        # 合并 possible_answer
        possible_path = os.path.join(data_path, "possible_answer", data_type)
        if os.path.exists(possible_path):
            with open(possible_path, "r") as f:
                possible_data = [json.loads(line.strip()) for line in f if line.strip()]
            possible_dict = {d["id"]: d["ground_truth"] for d in possible_data}
            for item in type_results:
                if item["id"] in possible_dict:
                    item["golden_answers"] = possible_dict[item["id"]]
        dataset.extend(type_results)

        type_datasets[data_type] = type_results

    
    
    df_row = pd.DataFrame(dataset)
    print(df_row.head())
    print(f"✅ 数据汇总完毕，共 {len(df_row)} 条样本")
    # print(df_row.groupby('source_type')['split'].value_counts())
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)


    def parallel_apply(df, func, max_workers=8):
        train_data = []
        results = [None] * len(df)
        fusion_results = [None] * len(df)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(func, row): i for i, row in df.iterrows()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
                i = futures[future]
                try:
                    results[i], fusion_results[i] = future.result()
                except Exception as e:
                    print(f"❌ Error processing row {i}: {e}")
        for result in results:
            if result is None:
                continue
            involved_classes = result["involved_classes"]
            
            for idx, item in enumerate(result["messages"]):
                if item["role"] == "assistant":
                    prompt = result["messages"][:idx]
                    
                    response = item["content"]
                    train_data.append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "involved_classes": involved_classes
                        }
                    )
        
        return train_data # results


    def apply_process_row(row):
        return process_single_row(row, row_index=row.name)
    
    # ✅ 并行执行
    results = parallel_apply(df_row, apply_process_row, max_workers=100)

    # ✅ 转为 DataFrame
    df_processed = pd.DataFrame(results)
    
    # Save processed DataFrame
    train_output_file_path = os.path.join(local_save_dir, f"train.parquet")
    df_processed.to_parquet(train_output_file_path, index=False)
    print(f"✅ Saved {len(df_processed)} processed rows to {train_output_file_path}")
    # logger.info(f"Saved {len(df_processed)} processed rows to {train_output_file_path}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")

def process_reasoning_row(row, row_index):
    prompt = row["prompt"].tolist()
    response = row["response"]
    involved_classes = row["involved_classes"]
        
    assert "<think>" not in response
    if "<think>" not in response:
        
        prompt_content = prompt.copy()
        prompt_content.append({"role": "user", "content": f"The correct answer for above dialogue is: {response}. Please generate the thinking process and the action or response!"}) # You need to generate the thinking process for the correct answer. You are not allowed to generate any other content and only the thinking process.
        
        response_content = client.chat.completions.create( # 
            model=MODEL_NAME,
            messages=prompt_content, 
            temperature=0.7,
            max_tokens=8192
        ).choices[0].message.content
        print(response_content)
        r = response_content.split("</think>")[-1]
        response_content = response_content.split("</think>")[0].replace("<think>", "")
        # assert _extract_tool_calls(r) == _extract_tool_calls(response), f"Tool calls in response are not the same: {r} != {response}"
        response = "<think>" + response_content.strip() + "</think>\n" + response
    return prompt, response, involved_classes
            
def process_reasoning(path, local_save_dir, max_workers=100):
    train_data = []
    df = pd.read_parquet(path)
    print(df.head())
    print(f"✅ 数据汇总完毕，共 {len(df)} 条样本")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_reasoning_row, row, row_index=row.name): i
            for i, (idx, row) in enumerate(df.iterrows())
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                prompt, response, involved_classes = future.result()
                train_data.append({"prompt": prompt, "response": response, "involved_classes": involved_classes})
            except Exception as e:
                print(f"❌ Error in task: {e}")

    df_processed = pd.DataFrame(train_data)

    train_output_file_path = os.path.join(local_save_dir, "train_reasoning.parquet")
    df_processed.to_parquet(train_output_file_path, index=False)
    print(f"✅ Saved {len(df_processed)} processed rows to {train_output_file_path}")
    

def process_candidate_row(row, row_index=None):
    prompt = row["prompt"].tolist()
    response = row["response"]
    involved_classes = row["involved_classes"]
        
    candidate_plans = []
    prompt_content = prompt 
    response_content = client.chat.completions.create( # 
        model=MODEL_NAME,
        messages=prompt_content, 
        temperature=1,
        max_tokens=8192,
        n=4,
    )
    candidate_action = []
    seen_calls = set()
    num = 0
    have_ground_truth = False
    ground_truth = {} if "<tool_call>" not in response else _extract_tool_calls(response)[0]
    candidate_plans = []
    for choice in response_content.choices:
        response_content = choice.message.content
        reasoning_content = response_content.split("</think>")[0].replace("<think>", "").strip()
        response_content = response_content.split("</think>")[-1]
        tool_call = _extract_tool_calls(response_content)
        candidate_plans.append({
            "thought": reasoning_content,
            "model_response": response_content,
            "tool_call": tool_call[0] if len(tool_call) > 0 else {}
        })
    
    for plan in candidate_plans:
        tool_call = plan.get("tool_call", {})
        if tool_call != {}:
            assert "arguments" in tool_call, f"Tool call {tool_call} does not have arguments"
        if tool_call == ground_truth:
            have_ground_truth = True
        tool_name = tool_call.get("name", None)
        parameters = json.dumps(tool_call.get("arguments", {}), sort_keys=True)
        
        
        key = f"{tool_name}:{parameters}"
        if key not in seen_calls:
            seen_calls.add(key)
            candidate_action.append({
                "thought": plan["thought"],
                "action": tool_call if tool_call != {} else plan["model_response"]
            })
            num += 1
    if not have_ground_truth:
        thought = response.split("</think>")[0].replace("<think>", "").strip()
        action = response.split("</think>")[-1]
        candidate_action.append({
            "thought": thought, 
            "action": action
        })
    candidate_str = ""
    for index, item in enumerate(candidate_action):
        candidate_str += f"\n### Candidate plans {index + 1}:\n"
        for k, v in item.items():
            candidate_str += f" - {k}: {str(v)}\n"
    ground_t = response.split("</think>")[-1] if ground_truth != {} else "dirctly answer to the user"
    prompt_candidate = prompt.copy()
    prompt_candidate.append({"role": "user", "content": decide_tool_calling_prompt.format(candidate_plans=candidate_str) + f".\nThe correct answer for above dialogue is:\n\n" + ground_t + "\n\nYour optimal tool call should be consistent with the correct answer."})
    
    response_candidate = client.chat.completions.create( # 
        model=MODEL_NAME,
        messages=prompt_candidate, 
        temperature=0.7,
        max_tokens=8192
    ).choices[0].message.content
    if "</think>" in response_candidate:
        fusion_result_o = response_candidate.split("</think>")[-1].strip()
        match = re.search(r'```json(.*?)```', fusion_result_o, re.DOTALL)
        if match:
            fusion_result = match.group(1).strip()
        else:
            fusion_result = fusion_result_o
        fusion_result = json.loads(fusion_result)
        tool_calls = fusion_result.get("optimal_tool_call", {})
        optimal_plan = fusion_result.get("optimal_plan", "")
        if ground_truth == {}:
            assert tool_calls["name"] == "response_to_user", f"Tool call name is response_to_user, but {tool_calls['name']}"
        else:
            fusion_result_call = tool_calls
            assert _extract_tool_calls(response)[0] == {"name": fusion_result_call["name"], "arguments": fusion_result_call["parameters"]}, f"Tool call ground truth is {_extract_tool_calls(response)}, but {fusion_result_call}"
            
    return prompt + [{"role": "user", "content": decide_tool_calling_prompt.format(candidate_plans=candidate_str)}], response_candidate, involved_classes

def process_candidate(path, output_path, max_workers=200):
    train_data = []
    df = pd.read_parquet(path)
    print(df.head())
    print(f"✅ 数据汇总完毕，共 {len(df)} 条样本")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_candidate_row, row, row_index=row.name): i
            for i, (idx, row) in enumerate(df.iterrows())
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                prompt, response, involved_classes = future.result()
                train_data.append({"prompt": prompt, "response": response, "involved_classes": involved_classes})
            except Exception as e:
                print(f"❌ Error in task: {e}")

    df_processed = pd.DataFrame(train_data)

    # train_output_file_path = os.path.join(local_save_dir, "train_candidate.parquet")
    df_processed.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(df_processed)} processed rows to {output_path}")
    return
    
        
def _extract_tool_calls(input_string):
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    matches = re.findall(pattern, input_string, re.DOTALL)

    # Process matches into a list of dictionaries
    result = []
    for match in matches:
        try:
            match = json.loads(match)
            result.append(match)
        except Exception as e:
            pass
    return result

def add_content(user_message, summary_result, involved_classes):
    # assert len(user_message) == 1
    content = user_message # ["content"]
    content_question = copy.deepcopy(content)
    pre_enhanced_content = (
                    "Before answering the user's question above, please first review the following related experiences:\n\n"
                )
    
    
    num = 0
    enhance_content = ""
    pattern = []
    question = []
    question = top_k_similar_questions_reflection_and_summary(summary_result, content, 3, 0.5, involved_classes,  True)
    for idx, question_item in enumerate(reversed(question)):
        information = summary_result[question_item]
        correct_response = information["answer"]
        summary = information["summary"]
        # Format the list of correct responses
        correct_response_str = "\n".join(
            [f"- {resp}" for resp in correct_response]
        )
        pre_enhanced_content += (
            f"### Example {idx + 1}\n"
            f"**Question:** {question_item}\n\n"
            f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
        )
        
    if len(question) > 0 or len(pattern) > 0:
        content += (
            "\n\n"
            + pre_enhanced_content
            + "\n**Note**: You are not required to reference the information or examples above "
            "if they are not directly relevant to the current user question. "
            "Analyze the problem carefully, decide whether the retrieved information is useful, "
            "and always apply reasoning before making any tool calls."
            + "\nYour actions must be based on the information given by the current user. "
            "You can not make up data, nor can you refer to examples that will cause you to act beyond the current information.\n"
            + "You need to determine the difference between your question and the question in retrieval examples."
            + f"\nAttention the user question at current turn is: \n{content_question}"
        )
    return content

def add_retrieval(path, output_path):
    with open("/home/yphao/Experience_Tool/test/berkeley-function-call-leaderboard/Experience/BFCL_v4_multi_turn_base_training_summary_with_embedding.json", "r") as f:
        summary_result = json.load(f)
    train_data = []
    df = pd.read_parquet(path)
    print(df.head())
    print(f"✅ 数据汇总完毕，共 {len(df)} 条样本")
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        response = row["response"]
        involved_classes = row["involved_classes"]
        for idx, item in enumerate(prompt):
            if item["role"] == "user" and "<tool_response>" not in item["content"]:
                item["content"] = add_content(item["content"], summary_result, involved_classes)

        train_data.append({"prompt": prompt, "response": response})
    df_processed = pd.DataFrame(train_data)

    df_processed.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(df_processed)} processed rows to {output_path}")
    


def get_rl_dataset(path, save_path, data_source):
    train_data = []
    df = pd.read_parquet(path)
    print(df.head())
    print(f"✅ 数据汇总完毕，共 {len(df)} 条样本")
    for idx, row in df.iterrows():
        prompt = row["prompt"].tolist()
        assert type(prompt) == list
        response = row["response"]
        assert type(response) == str
        ground_truth = response
        
        train_data.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "tool",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    
                },
            }
        )
    
    df_processed = pd.DataFrame(train_data)

    df_processed.to_parquet(save_path, index=False)
    print(f"✅ Saved {len(df_processed)} processed rows to {save_path}")

from pathlib import Path

def split_sft_rl(
    in_file: str,
    out_sft: str,
    out_rl: str,
    ratio: float = 0.5,
    seed: int = 42,
):
    df = pd.read_parquet(in_file)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * ratio)

    df_sft = df.iloc[:split_idx]
    df_rl = df.iloc[split_idx:]

    Path(out_sft).parent.mkdir(parents=True, exist_ok=True)
    df_sft.to_parquet(out_sft, index=False)
    df_rl.to_parquet(out_rl, index=False)

    print(f"[Split] {in_file}: SFT={len(df_sft)}, RL={len(df_rl)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get fcl training data and convert")
    parser.add_argument(
        "--input_dir", default="", help="Input directory to load the Parquet files."
    )
    parser.add_argument(
        "--local_dir",
        default="./data/training_data/", 
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()
    # main()
    input_dir = args.local_dir
    
    base = args.local_dir

    split_sft_rl(
        os.path.join(base, "train_reasoning.parquet"),
        os.path.join(base, "train_reasoning_sft.parquet"),
        os.path.join(base, "train_reasoning_rl_base.parquet")
    )

    add_retrieval(
        os.path.join(base, "train_reasoning.parquet"),
        os.path.join(base, "train_reasoning_retrieval.parquet")
    )

    split_sft_rl(
        os.path.join(base, "train_reasoning_retrieval.parquet"),
        os.path.join(base, "train_reasoning_sft_retrieval.parquet"),
        os.path.join(base, "train_reasoning_rl_base_retrieval.parquet")
    )

    get_rl_dataset(
        os.path.join(base, "train_reasoning_rl_base_retrieval.parquet"),
        os.path.join(base, "train_reasoning_rl_retrieval.parquet"),
        "function"
    )
    
    dataset = pd.read_parquet(os.path.join(base, "train_reasoning_rl_retrieval.parquet"))

    new_indices = []
    new_rows = []

    for _, data in dataset.iterrows():
        tokens = tokenizer.apply_chat_template(
            data["prompt"],
            add_generation_prompt=True,
            tokenize=True
        )

        if len(tokens) <= 8192:
            new_rows.append(data)
            new_indices.append(data["extra_info"]["index"])

    df_new = pd.DataFrame(new_rows)
    df_new.to_parquet(os.path.join(base, "train_reasoning_rl_retrieval.parquet"), index=False)


    




    dataset = pd.read_parquet(os.path.join(base, "train_reasoning_sft_retrieval.parquet"))
    

    new_indices = []
    new_rows = []

    for idx, data in dataset.iterrows():
        tokens = tokenizer.apply_chat_template(
            data["prompt"],
            add_generation_prompt=True,
            tokenize=True
        )

        if len(tokens) <= 8192:
            new_rows.append(data)
            new_indices.append(idx)

    df_new = pd.DataFrame(new_rows)
    df_new.to_parquet(os.path.join(base, "train_reasoning_sft_retrieval.parquet"), index=False)
