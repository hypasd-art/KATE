from typing import Any, Dict, List
import copy
import re
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override
import os
import json
from openai import OpenAI
import time
from bfcl_eval.model_handler.local_inference.MA_prompt import * 
from bfcl_eval.model_handler.utils import (
    top_k_similar_questions_reflection_and_summary,
    top_k_similar_questions_intent,
)
from bfcl_eval.model_handler.utils import *


class QwenHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.is_fc_model = False

    @override
    def _format_prompt(self, messages, function):
        """
        "chat_template":
        {%- if tools %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0].role == 'system' %}
                {{- messages[0].content + '\n\n' }}
            {%- endif %}
            {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
        {%- else %}
            {%- if messages[0].role == 'system' %}
                {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
        {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
                {%- set ns.multi_step_tool = false %}
                {%- set ns.last_query_index = index %}
            {%- endif %}
        {%- endfor %}
        {%- for message in messages %}
            {%- if message.content is string %}
                {%- set content = message.content %}
            {%- else %}
                {%- set content = '' %}
            {%- endif %}
            {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
                {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in content %}
                        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- if loop.index0 > ns.last_query_index %}
                    {%- if loop.last or (not loop.last and reasoning_content) %}
                        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
                    {%- else %}
                        {{- '<|im_start|>' + message.role + '\n' + content }}
                    {%- endif %}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if (loop.first and content) or (not loop.first) %}
                            {{- '\n' }}
                        {%- endif %}
                        {%- if tool_call.function %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {{- '<tool_call>\n{"name": "' }}
                        {{- tool_call.name }}
                        {{- '", "arguments": ' }}
                        {%- if tool_call.arguments is string %}
                            {{- tool_call.arguments }}
                        {%- else %}
                            {{- tool_call.arguments | tojson }}
                        {%- endif %}
                        {{- '}\n</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
                    {{- '<|im_start|>user' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- content }}
                {{- '\n</tool_response>' }}
                {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
        {%- endif %}
        """
        formatted_prompt = ""

        if messages[0]["role"] == "system":
            formatted_prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]

                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }


    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses"],
                "reasoning_content": model_response_data.get("reasoning_content", ""),
            }
        )
        return inference_data
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
    ast_parse,
    ReturnFormat,
)


        
from bfcl_eval.model_handler.api_inference.hf_multi_agent import TaskProcessingSystem
from concurrent.futures import ThreadPoolExecutor, as_completed
  

    


    

class MA_QwenEnhanceHandler(QwenHandler):
    def __init__(self, model_name, temperature, method, skip_first_example=False, information_dict=None, reasoning_enhance=False, MA=False, MA_prompt=None, sample_num=1, trajectory_retrieval=False, fusion_method="majority", fusion_prompt="single_turn", preprocess_fusion=False) -> None:
        super().__init__(model_name, temperature)
        self.model_name_huggingface = model_name.replace("-Enhance", "").replace("-train", "").replace("-retrieval", "").replace("-rl", "")

        self.method = method.split(",")
        self.skip_first_example = skip_first_example
        self.reasoning_enhance = reasoning_enhance
        self.retrieval_hit_time = 0
        self.total_time = 0
        self.prompt = False

        self.trajectory_retrieval = trajectory_retrieval
        self.MA = MA
        self.MA_prompt = MA_prompt
        self.sample_num = sample_num

        self.max_output_tokens = 3000
        
        self.thread_pool = ThreadPoolExecutor(max_workers=100)
        self.same_action = 0
        self.total_action = 0
        self.fusion_method = fusion_method
        self.fusion_prompt = fusion_prompt

        self.preprocess_fusion = preprocess_fusion
        self.inference_tokens = 0
        
        if "summary" in self.method or "trajectory" in self.method:
            with open(information_dict, "r") as f:
                self.summary_result = json.load(f)
        if "intent" in self.method or "intent_summary" in self.method:
            with open("./Experience/intent_pattern.json", "r") as f:
                self.intent = json.load(f)
        
        
    @override
    def decode_execute(self, result, has_tool_call_tag):
        function_call_list = result
        if type(function_call_list) == str:
            return []
        if type(function_call_list) == dict:
            function_call_list = [function_call_list]
        execution_list = []
        for function_call in function_call_list:
            execution_list.append(
                f"{function_call['name']}({','.join([f'{k}={repr(v)}' for k,v in function_call['parameters'].items()])})"
            )
        return execution_list

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result in execution_results:
            inference_data["message"].append(
                {"role": "tool", "content": execution_result}
            )

        return inference_data

    
    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        # only one tool call is included
        if model_response_data.get("tool_calls") != {}:
            try:
                tool_calls = [{"name": model_response_data.get("tool_calls", {})["name"], "parameters": model_response_data.get("tool_calls", {})["parameters"]}] # parameter
            except:
                print(model_response_data)
                raise NameError("tool_calls")
        else:
            tool_calls = []
        # the content is empty and the content is store in tool_calls and model_response
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": model_response_data.get("reasoning_content", ""),
                "model_response": model_response_data.get("model_responses", ""),
                "tool_calls": tool_calls
            }
        )
        return inference_data
        
    @override
    def _query_prompting(self, inference_data: dict):
        
        start_time = time.time()
        message: list[dict] = inference_data["message"]
        function = inference_data["function"]
        # not include the reasoning content in parallel decoding
        # the temperature is set to 1 in parallel decoding
        if self.MA:
            plan_futures = [
                self.thread_pool.submit(self.subtask_planner_FC, tools=function, messages=message, prompt=state_prompt, temperature=0),
                self.thread_pool.submit(self.subtask_planner_FC, tools=function, messages=message, prompt=reflection_prompt, temperature=0),
                self.thread_pool.submit(self.subtask_planner_FC, tools=function, messages=message, prompt=intent_prompt, temperature=0),
                self.thread_pool.submit(self.subtask_planner_FC, tools=function, messages=message, temperature=0),
            ]
            candidate_plans = [future.result() for future in plan_futures]
            
            self.total_action += 1
            
            optimal_plan, response, tool_calls = self.decision_fusion_and_tool_validator_new(candidate_plans, message, function, self.fusion_method)
            api_response = {"reasoning_content": "", "model_response": response, "tool_calls": tool_calls}
        elif self.sample_num > 1:
            result = self.subtask_planner_FC(tools=function, messages=message, temperature=1, num=self.sample_num)
            candidate_plans = result
            self.total_action += 1
            
            optimal_plan, response, tool_calls = self.decision_fusion_and_tool_validator_new(candidate_plans, message, function, self.fusion_method)
            api_response = {"reasoning_content": "", "model_response": response, "tool_calls": tool_calls}
        else:
            if self.MA_prompt == "default":
                result = self.subtask_planner_FC(tools=function, messages=message, temperature=self.temperature)
            elif self.MA_prompt == "state":
                result = self.subtask_planner_FC(tools=function, messages=message, prompt=state_prompt, temperature=self.temperature)
            elif self.MA_prompt == "reflection":
                result = self.subtask_planner_FC(tools=function, messages=message, prompt=reflection_prompt, temperature=self.temperature)
            elif self.MA_prompt == "intent":
                result = self.subtask_planner_FC(tools=function, messages=message, prompt=intent_prompt, temperature=self.temperature)
            api_response = {"reasoning_content": result["thought"], "model_response": result["model_response"], "tool_calls": result["tool_call"]}
        
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response["tool_calls"] if api_response["tool_calls"] != {} else api_response["model_response"], #   {} optimal_plan
            "reasoning_content": api_response["reasoning_content"],
            "tool_calls": api_response["tool_calls"],
            "input_token": 0,
            "output_token": 0,
        }

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        inference_data = {}
        inference_data["message"] = []
        inference_data["involved_classes"] = test_entry["involved_classes"]
        inference_data["function"] = functions
        if self.prompt:
            test_entry_id: str = test_entry["id"]

            test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                test_entry["question"][0], functions, test_entry_id
            )

        return inference_data 


    def _get_intent(
        self, inference_data: dict, content: str
    ) -> dict:
        message: list[dict] = copy.deepcopy(inference_data["message"])
        tools = inference_data["function"]
        message = [{"role": "user", "content": content + "\n\n" + "Please summarize the user's intent in one sentence."}]
        prompt = self._format_prompt(message, tools)
        api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=prompt,
                max_tokens=1000,
                timeout=72000,  # Avoid timeout errors
            )
        model_response = api_response.choices[0].text
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")
        else:
            cleaned_response = model_response
        
        return cleaned_response

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        if self.prompt:
            assert len(first_turn_message) == 2
            content = first_turn_message[1]["content"]
        else:
            assert len(first_turn_message) == 1
            content = first_turn_message[0]["content"]
        content_question = copy.deepcopy(content) 
        pre_enhanced_content = (
                        "Before answering the user's question above, please first review the following related experiences:\n\n"
                    )
        involved_classes = inference_data["involved_classes"]
                    
        self.total_time += 1
        num = 0
        enhance_content = ""
        pattern = []
        question = []
        if "intent" in self.method or "intent_summary" in self.method:
            intent = self._get_intent(inference_data, content)
            pattern = top_k_similar_questions_intent(self.intent, intent, 1, 0.5, involved_classes) # _result
            if len(pattern) > 0:
                pre_enhanced_content += f"The user's intent is {intent}\nThere are some behavior pattern for you to reference:\n"
                for item in pattern:
                    if "intent" in self.method:
                        pre_enhanced_content += f"**Pattern of {item['intent']}**: {json.dumps(item['pattern']['step'])}"
                    elif "intent_summary" in self.method:
                        pre_enhanced_content += f"**Pattern Summary of {item['intent']}**: {item['pattern']['summary']}" # ["summary"]  \n**Step**: {json.dumps(item['step'])}\n
        if self.trajectory_retrieval:
            question = top_k_similar_questions_reflection_and_summary(self.summary_result, content, 3, 0.5, involved_classes,  self.skip_first_example)


            
            for idx, question_item in enumerate(reversed(question)):
                information = self.summary_result[question_item]
                correct_response = information["answer"]
                summary = information["summary"]
                # Format the list of correct responses
                correct_response_str = "\n".join(
                    [f"- {resp}" for resp in correct_response]
                )
                if "summary" not in self.method and "trajectory" in self.method:
                    pre_enhanced_content += (
                        f"### Example {idx + 1}\n"
                        f"**Question:** {question_item}\n\n"
                        f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
                    )
                elif "summary" in self.method and "trajectory" not in self.method:
                    pre_enhanced_content += (
                        f"### Example {idx + 1}\n"
                        f"**Question:** {question_item}\n\n"
                        f"**Analysis & Advice:**\n{summary}\n\n"
                    )
                elif "summary" in self.method and "trajectory" in self.method:
                    pre_enhanced_content += (
                        f"### Example {idx + 1}\n"
                        f"**Question:** {question_item}\n\n"
                        f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
                        f"**Analysis & Advice:**\n{summary}\n\n"
                    )
            
        if len(question) > 0 or len(pattern) > 0:
            self.retrieval_hit_time += 1
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
        
        
        
        if self.prompt:
            first_turn_message[1]["content"] = content
        else:
            first_turn_message[0]["content"] = content
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        assert len(user_message) == 1
        content = user_message[0]["content"]
        content_question = copy.deepcopy(content)
        pre_enhanced_content = (
                        "Before answering the user's question above, please first review the following related experiences:\n\n"
                    )
        
        involved_classes = inference_data["involved_classes"]
        
        self.total_time += 1
        num = 0
        enhance_content = ""
        pattern = []
        question = []
        if "intent" in self.method or "intent_summary" in self.method:
            intent = self._get_intent(inference_data, content)
            pattern = top_k_similar_questions_intent(self.intent, intent, 1, 0.5, involved_classes) 
            if len(pattern) > 0:
                pre_enhanced_content += f"The user's intent is {intent}\nThere are some behavior pattern for you to reference:\n"
                for item in pattern:
                    if "intent" in self.method:
                        pre_enhanced_content += f"**Pattern of {item['intent']}**: {json.dumps(item['pattern']['step'])}"
                    elif "intent_summary" in self.method:
                        pre_enhanced_content += f"**Pattern Summary of {item['intent']}**: {item['pattern']['summary']}" 
        if self.trajectory_retrieval:
            question = top_k_similar_questions_reflection_and_summary(self.summary_result, content, 3, 0.5, involved_classes,  self.skip_first_example)
            


            
            
            for idx, question_item in enumerate(reversed(question)):
                information = self.summary_result[question_item]
                correct_response = information["answer"]
                summary = information["summary"]
                # Format the list of correct responses
                correct_response_str = "\n".join(
                    [f"- {resp}" for resp in correct_response]
                )
                if "summary" not in self.method and "trajectory" in self.method:
                    pre_enhanced_content += (
                        f"### Example {idx + 1}\n"
                        f"**Question:** {question_item}\n\n"
                        f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
                    )
                elif "summary" in self.method and "trajectory" not in self.method:
                    pre_enhanced_content += (
                        f"### Example {idx + 1}\n"
                        f"**Question:** {question_item}\n\n"
                        f"**Analysis & Advice:**\n{summary}\n\n"
                    )
                elif "summary" in self.method and "trajectory" in self.method:
                    pre_enhanced_content += (
                        f"### Example {idx + 1}\n"
                        f"**Question:** {question_item}\n\n"
                        f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
                        f"**Analysis & Advice:**\n{summary}\n\n"
                    )
            
        if len(question) > 0 or len(pattern) > 0:
            self.retrieval_hit_time += 1
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
        # print(self.retrieval_hit_time, self.total_time, self.retrieval_hit_time / self.total_time)
        
        '''print(f"Content: {content}")'''
        '''print(f"Content: {content}")'''
        user_message[0]["content"] = content
        inference_data["message"].extend(user_message)
        return inference_data

    def decide_prompt_single(self, messages, function, candidate_plans, angle=None):
        formatted_prompt = ""
        candidate_str = ""
        for idx, item in enumerate(candidate_plans):
            candidate_str += f"### Candidate plans {idx + 1}:\n"
            for k, v in item.items():
                candidate_str += f" - {k}: {str(v)}\n"
        history_messages = ""
        for idx, item in enumerate(messages):
            role = item["role"]
            content = item["content"]
            if role == "assistant":
                if item["tool_calls"] == []:
                    content = item["model_response"]
                for tool_call in item["tool_calls"]:
                    if (tool_call == item["tool_calls"][0] and content) or tool_call != item["tool_calls"][0]:
                        content += "\n"
                    
                    if "function" in tool_call:
                        tool_call = tool_call["function"]
                    
                    content += '<tool_call>\n{"name": "'
                    content += tool_call["name"]
                    content += '", "arguments": '
                    
                    if isinstance(tool_call["parameters"], str):
                        content += tool_call["parameters"]
                    else:
                        content += json.dumps(tool_call["parameters"])
                    
                    content += "}\n</tool_call>"
            history_messages += f"{role}:\n{content}\n\n"
        function_str = ""
        for idx, item in enumerate(function):
            function_str += f"\n{json.dumps(item)}"
        if angle is None:
            formatted_prompt = "<|im_start|>user\n" + decide_tool_calling_prompt_one_turn.format(history_messages=history_messages, available_tools=function_str, candidate_plans=candidate_str) + "<|im_end|>\n" # json.dumps(candidate_plans)
        else:
            decide_prompt = decide_prompt_dict_one_turn[angle]
            formatted_prompt = "<|im_start|>user\n" + decide_prompt.format(history_messages=history_messages, available_tools=function_str, candidate_plans=candidate_str) + "<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    def decide_prompt(self, messages, function, candidate_plans, angle=None):
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                assert content == ""
                if message["tool_calls"] == []:
                    content = message["model_response"]
                formatted_prompt += f"<|im_start|>{role}\n{content}"
                    
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        
                        if isinstance(tool_call["parameters"], str):
                            formatted_prompt += tool_call["parameters"]
                        else:
                            formatted_prompt += json.dumps(tool_call["parameters"])
                        
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"
        candidate_str = ""
        for idx, item in enumerate(candidate_plans):
            candidate_str += f"### Candidate plans {idx + 1}:\n"
            for k, v in item.items():
                candidate_str += f" - {k}: {str(v)}\n"
        if angle is None:
            formatted_prompt += "<|im_start|>user\n" + decide_tool_calling_prompt.format(candidate_plans=candidate_str) + "<|im_end|>\n" 
        else:
            decide_prompt = decide_prompt_dict[angle]
            formatted_prompt += "<|im_start|>user\n" + decide_prompt.format(candidate_plans=candidate_str) + "<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt
        

    def fc_prompt(self, messages, function, intent_state=None, prompt=None):
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"
            
            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            
            formatted_prompt += '<|im_end|>\n'

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                assert content == ""
                if message["tool_calls"] == []:
                    content = message["model_response"]
                if "reasoning" in self.method:
                    reasoning_content = ""
                    if "reasoning_content" in message and message["reasoning_content"]:
                        reasoning_content = message["reasoning_content"]

                    elif "</think>" in content:
                        parts = content.split("</think>")
                        reasoning_content = (
                            parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                        )
                        content = parts[-1].lstrip("\n")

                    if idx > last_query_index:
                        if idx == len(messages) - 1 or reasoning_content:
                            formatted_prompt += (
                                f"<|im_start|>{role}\n<think>\n"
                                + reasoning_content.strip("\n")
                                + f"\n</think>\n\n"
                                + content.lstrip("\n")
                            )
                        else:
                            formatted_prompt += f"<|im_start|>{role}\n{content}"
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
                    
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        
                        if isinstance(tool_call["parameters"], str):
                            formatted_prompt += tool_call["parameters"]
                        else:
                            formatted_prompt += json.dumps(tool_call["parameters"])
                        
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"
        if prompt is not None:
            formatted_prompt += "<|im_start|>user\n" + prompt + "\n<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    def _extract_tool_calls(self, input_string):
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

    def _call_llm_api(self, messages: list[dict], tools: list[dict] = None, prompt: str = None, temperature: float = None, num: int = 1):
        if prompt is not None:
            
            response = self.client.completions.create(
                model=self.model_name_huggingface,
                prompt=prompt,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_output_tokens,
                n=num,
            )
            self.inference_tokens += response.usage.total_tokens
            
            return response.model_dump()
        response = self.client.chat.completions.create(
            model=self.model_name_huggingface,
            messages=messages,
            tools=tools if tools else None,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_output_tokens,
            n=num,
        )
        self.inference_tokens += response.usage.total_tokens
        return response.model_dump()  
        
        
    def subtask_planner_FC(self, tools: List = None, messages: List = None, prompt: str = None, temperature = None, num = 1) -> Dict[str, Any]:
        prompt = self.fc_prompt(messages, tools, prompt=prompt)
        
        while True:
            try:
                
                result = self._call_llm_api(messages=messages, prompt=prompt, tools=tools, temperature=temperature, num=num) 
                
                if num > 1:
                    candidate_plans = []
                    for i in range(len(result["choices"])):
                        llm_content = result["choices"][i]["text"]
                        reasoning_content = ""
                        if "</think>" in llm_content:
                            parts = llm_content.split("</think>")
                            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                            llm_content = parts[-1].lstrip("\n")
                        
                        tool_calls = self._extract_tool_calls(llm_content)
                        if len(tool_calls) > 0:
                            tool_calls = tool_calls[0]
                            # Build complete plan data
                            plan_data = {
                                "thought": reasoning_content,  
                                "if_calling": True,
                                "tool_call": {"name": tool_calls["name"], "parameters": tool_calls["arguments"]},
                                "model_response": ""
                            }
                        else:
                            plan_data = {
                                "thought": reasoning_content, 
                                "if_calling": False,
                                "tool_call": {},
                                "model_response": llm_content
                            }
                        candidate_plans.append(plan_data)
                    return candidate_plans
                
                llm_content = result["choices"][0]["text"]
                
                reasoning_content = ""
                if "</think>" in llm_content:
                    parts = llm_content.split("</think>")
                    reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    llm_content = parts[-1].lstrip("\n")
                
                tool_calls = self._extract_tool_calls(llm_content)
                if len(tool_calls) > 0:
                    tool_calls = tool_calls[0]
                    # Build complete plan data
                    plan_data = {
                        "thought": reasoning_content,  
                        "if_calling": True,
                        "tool_call": {"name": tool_calls["name"], "parameters": tool_calls["arguments"]},
                        "model_response": ""
                    }
                else:
                    plan_data = {
                        "thought": reasoning_content, 
                        "if_calling": False,
                        "tool_call": {},
                        "model_response": llm_content
                    }
                return plan_data
                
            except Exception as e:
                print(f"\n\n!!!Error in function calling: \n\n{str(e)}\n\n")
                if "Request timed out" not in str(e):
                    print("Return empty action due to error.")
                else:
                    continue
                
                if num > 1:
                    return [
                                {
                            "thought": "",
                            "if_calling": False,
                            "model_response": "",
                            "tool_call": {},
                        }
                    ]
                return {
                    "thought": "",
                    "if_calling": False,
                    "model_response": "",
                    "tool_call": {},
                }

        
    def decision_fusion_and_tool_validator_new(self, candidate_plans: List[Dict], messages: List[Dict[str, str]], tools: List, fusion_method: str) -> Dict[str, Any]:
        from collections import Counter
        # 1. 判断 candidate_plans 是否所有工具调用相同
        same_call = True
        if len(candidate_plans) > 1:
            tool_call_begin = candidate_plans[0]["tool_call"]
            for item in candidate_plans:
                tool_call = item.get("tool_call", {})
                if tool_call != tool_call_begin:
                    same_call = False
                    break
        if same_call:
            self.same_action += 1
            if candidate_plans[0]["if_calling"]:
                return candidate_plans[0]["thought"], candidate_plans[0]["model_response"], candidate_plans[0]["tool_call"]
            return candidate_plans[0]["thought"], candidate_plans[0]["model_response"], {}

        # 2. 生成去重后的 candidate_action
        candidate_action = []
        seen_calls = set()
        for plan in candidate_plans:
            tool_call = plan.get("tool_call", {})
            tool_name = tool_call.get("name", None)
            parameters = json.dumps(tool_call.get("parameters", {}), sort_keys=True)
            key = f"{tool_name}:{parameters}"
            if key not in seen_calls:
                seen_calls.add(key)
                if self.reasoning_enhance:
                    candidate_action.append({
                        "thought": plan["thought"],
                        "action": tool_call if tool_call != {} else plan["model_response"]
                    })
                else:
                    candidate_action.append({
                        "action": tool_call if tool_call != {} else plan["model_response"]
                    })

        print(f"len(candidate_plans): {len(candidate_plans)}, unique actions: {len(candidate_action)}, same_action: {self.same_action}/{self.total_action}={self.same_action/self.total_action}")
        if self.fusion_method == "majority":
            tool_counter = Counter()
            for plan in candidate_plans:
                tool_call = plan.get("tool_call", {})
                key = json.dumps(tool_call) 
                tool_counter[key] += 1
            tool_call = tool_counter.most_common(1)[0][0]
            print(f"tool_counter: {tool_counter}\ntool call:{tool_call}")
            if json.loads(tool_call) == {}:
                return "", candidate_plans[0]["model_response"], {}
            return "", "", json.loads(tool_call)
        elif self.fusion_method == "critic":
            message_prompt = copy.deepcopy(messages)
            if self.preprocess_fusion:
                tool_counter = Counter()
                for plan in candidate_plans:
                    tool_call = plan.get("tool_call", {})
                    key = json.dumps(tool_call) 
                    tool_counter[key] += 1
                candidate_act = []
                has_response = False
                for item in candidate_action:
                    for key, count in tool_counter.most_common(2):
                        if key == "{}":
                            has_response = True
                        if item["action"] == json.loads(key):
                            candidate_act.append(item)
                if has_response:
                    for item in candidate_action:
                        if not isinstance(item["action"], dict):
                            candidate_act.append(item)
                print(f"candidate_act: {len(candidate_act)}/{len(candidate_action)} {candidate_act}")
                candidate_action = candidate_act
            if self.fusion_prompt == "single_turn":
                prompt = self.decide_prompt_single(message_prompt, tools, candidate_action)
            else:
                prompt = self.decide_prompt(message_prompt, tools, candidate_action)
            attempt = 0
            while attempt < 3:
                try:
                    result = self._call_llm_api(messages, prompt=prompt)
                    result_text = result["choices"][0]["text"].strip()

                    # 提取 <think> 中的 reasoning 内容
                    reasoning_content = ""
                    if "</think>" in result_text:
                        parts = result_text.split("</think>")
                        reasoning_content = parts[0].split("<think>")[-1]
                        result_text = parts[-1].strip()

                    match = re.search(r'```json(.*?)```', result_text, re.DOTALL)
                    if match:
                        fusion_result = match.group(1).strip()
                    else:
                        fusion_result = result_text
                    fusion_result = json.loads(fusion_result)
                    tool_calls = fusion_result.get("optimal_tool_call", {})
                    optimal_plan = fusion_result.get("optimal_plan", "")
                    print(tool_calls)
                    if tool_calls["name"] == "response_to_user":
                        model_response = tool_calls["parameters"]["content"]
                        return "", model_response, {}
                    else:
                        return "", "", tool_calls
            
                except Exception as e:
                    attempt += 1
                    print(f"Attempt {attempt} failed: {e}")
            return candidate_plans[0]["thought"], candidate_plans[0]["model_response"], candidate_plans[0]["tool_call"]
        elif self.fusion_method == "critic_majority":
            # 3. 多角度 prompt 分析
            message_prompt = copy.deepcopy(messages)
            
            angles = ["normal", "function_and_parameters_check", "logic_check", "current_state_check"]
            tool_counter = Counter()
            thought_results = []
            response_text = ""

            for angle in angles:
                if self.fusion_prompt == "single_turn":
                    prompt = self.decide_prompt_single(message_prompt, tools, candidate_action, angle=angle)
                else:
                    prompt = self.decide_prompt(message_prompt, tools, candidate_action, angle=angle)
                attempt = 0
                while attempt < 3:
                    try:
                
                        result = self._call_llm_api(messages, prompt=prompt)
                        result_text = result["choices"][0]["text"].strip()

                        # 提取 <think> 中的 reasoning 内容
                        reasoning_content = ""
                        if "</think>" in result_text:
                            parts = result_text.split("</think>")
                            reasoning_content = parts[0].split("<think>")[-1]
                            result_text = parts[-1].strip()

                        match = re.search(r'```json(.*?)```', result_text, re.DOTALL)
                        if match:
                            fusion_result = match.group(1).strip()
                        else:
                            fusion_result = result_text
                        fusion_result = json.loads(fusion_result)
                        tool_calls = fusion_result.get("optimal_tool_call", {})
                        optimal_plan = fusion_result.get("optimal_plan", "")
                        if tool_calls["name"] == "response_to_user":
                            model_response = tool_calls["parameters"]["content"]
                            response_text = model_response
                            key = "content" # result_text
                            tool_counter[key] += 1
                        elif tool_calls["name"] != "response_to_user":
                            key = json.dumps(tool_calls, sort_keys=True)
                            tool_counter[key] += 1
                        thought_results.append((angle, reasoning_content, result_text))
                    except Exception as e:
                        attempt += 1
                        print(f"Attempt {attempt} for angle {angle} failed: {e}")
                        

            # 4. 选出现次数最多的工具调用
            if tool_counter:
                most_common_tool_json, count = tool_counter.most_common(1)[0]
                try:
                    final_tool_call = json.loads(most_common_tool_json)
                except:
                    assert response_text != ""
                    final_tool_call = {}
            else:
                # 如果都没有工具调用，返回空
                final_tool_call = {}
                return candidate_plans[0]["thought"], candidate_plans[0]["model_response"], candidate_plans[0]["tool_call"]

            print(f"Tool call counts: {tool_counter}")
            if final_tool_call == {}:
                assert response_text != ""
                return candidate_plans[0]["thought"], response_text, {} # ""
            return "", "", {"name": final_tool_call["name"], "parameters": final_tool_call["parameters"]}
