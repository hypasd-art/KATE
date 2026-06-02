import json
import os
import time
from typing import Any

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from overrides import override
from bfcl_eval.model_handler.api_inference.MA_prompt import * # react_prompt, planexec_prompt, intent_state_prompt, decision_fusion_and_tool_validator_prompt
from bfcl_eval.model_handler.utils import (
    top_k_similar_questions_reflection_and_summary,
    top_k_similar_questions_abstract, reasoning_enhance_prompt,
    top_k_similar_questions_intent,
)
from typing import Any, Dict, List
import re

class OpenAICompletionsHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.v3.cm/v1")

    def decode_ast(self, result, language, has_tool_call_tag):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        # print(result)
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "messages": message,
            "model": self.model_name.replace("-FC", ""),
            "temperature": self.temperature,
            "store": False,
        }

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
        except:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    def _add_reasoning_content_if_available_FC(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        # Original assistant message object (contains `reasoning_content` on DeepSeek).
        message = api_response.choices[0].message

        # Preserve tool_call information but strip the unsupported `reasoning_content` field before inserting into chat history.
        if getattr(message, "tool_calls", None):
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            response_data["model_responses_message_for_chat_history"] = assistant_message

        # If no tool_calls, we still need to strip reasoning_content.
        elif hasattr(message, "reasoning_content"):
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": message.content,
            }

        # Capture the reasoning trace so it can be logged to the local result file.
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}
        
        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.model_name,
            temperature=self.temperature,
            store=False,
        )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        # print(api_response.choices[0].message.content)
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data

    def _add_reasoning_content_if_available_prompting(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        message = api_response.choices[0].message
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content
            # Reasoning content should not be included in the chat history
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": str(response_data["model_responses"]),
            }

from bfcl_eval.model_handler.utils import (
    top_k_similar_questions_reflection_and_summary,
    top_k_similar_questions_abstract, reasoning_enhance_prompt,
    top_k_similar_questions_intent,
)
import copy
from dataclasses import dataclass, field
from typing import List, Any
import time

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    # ,
    # ,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
import time
import uuid
import json


    

from bfcl_eval.model_handler.api_inference.hf_multi_agent import TaskProcessingSystem
from concurrent.futures import ThreadPoolExecutor, as_completed
class MA_OpenAICompletionsEnhancedHandler(OpenAICompletionsHandler):
    def __init__(self, model_name: str, temperature: float, method, skip_first_example=False, information_dict=None, reasoning_enhance=False, MA=False, MA_prompt=None, sample_num=1, trajectory_retrieval=False, fusion_method=None, fusion_prompt=None):
        super().__init__(model_name, temperature)
        self.model_name_huggingface = model_name.replace("-Enhance", "")

        self.method = method.split(",")
        self.skip_first_example = skip_first_example
        self.reasoning_enhance = reasoning_enhance
        self.retrieval_hit_time = 0
        self.total_time = 0

        self.MA = MA
        self.MA_prompt = MA_prompt
        self.sample_num = sample_num
        self.trajectory_retrieval = trajectory_retrieval

        self.thread_pool = ThreadPoolExecutor(max_workers=100)
        self.same_action = 0
        self.total_action = 0
        
        self.fusion_method = fusion_method
        self.fusion_prompt = fusion_prompt
        
        if "summary" in self.method or "trajectory" in self.method:
            with open(information_dict, "r") as f:
                self.summary_result = json.load(f)
        if "intent" in self.method or "intent_summary" in self.method:
            with open("./Experience/intent_pattern.json", "r") as f:
                self.intent = json.load(f)

        self.fc_messages = []
    @override
    def decode_execute(self, result, has_tool_call_tag):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)
    
    def format_chat_tool_response(self, input_dict):
        """Convert {"model_response": str, "tool_call": str | dict} into ChatCompletion"""

        model_response = input_dict.get("model_response")
        tool_call = input_dict.get("tool_calls")

        tool_calls = None
        finish_reason = "stop"

        # 若存在工具调用，构造 tool_calls 结构
        if tool_call != {}:
            if isinstance(tool_call, str):
                tool_call = json.loads(tool_call)  # 若为字符串，则解析 JSON

            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=str(uuid.uuid4()),
                    type="function",
                    function=dict(
                        name=tool_call.get("name"),
                        arguments=json.dumps(tool_call.get("parameters"), ensure_ascii=False),
                    )
                )
            ]
            finish_reason = "tool_calls"

        # 构建 message
        msg = ChatCompletionMessage(
            role="assistant",
            content=model_response,
            tool_calls=tool_calls
        )

        choice = Choice(
            index=0,
            message=msg,
            finish_reason=finish_reason,
        )

        return ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.model_name.replace("-FC", "").replace("-Enhance", ""),
            choices=[choice],
            usage=CompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
        
    def decision_fusion_and_tool_validator(self, candidate_plans: List[Dict], inference_data) -> Dict[str, Any]:
        """
        Fusion multiple candidate plans into an optimal plan and validate tools
        """
        from collections import Counter
        messages = copy.deepcopy(inference_data["message"])
        same_call = True
        if len(candidate_plans) > 1:
            try:
                tool_call_begin = candidate_plans[0]["tool_call"]
            except:
                tool_call_begin = {}
            for item in candidate_plans:
                try:
                    tool_call = item["tool_call"]
                except:
                    tool_call = {}
                if tool_call != tool_call_begin:
                    same_call = False
                    break
        self.total_action += 1
        if same_call and tool_call_begin == {}:
            self.same_action += 1
            print(f"all plans are the same, return the first plan: {candidate_plans[0]}")
            return {"model_response": candidate_plans[0]["model_response"], "tool_calls": {}}
        if same_call:
            self.same_action += 1
            print(f"all plans are the same, return the first plan: {tool_call_begin}")
            return {"model_response": "", "tool_calls": tool_call_begin}
        candidate_action = []
        seen_calls = set()
        for plan in candidate_plans:
            tool_call = plan.get("tool_call", {})
            key = json.dumps(tool_call)
            if key not in seen_calls:
                seen_calls.add(key)
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
                return {"model_response": candidate_plans[0]["model_response"], "tool_calls": {}}
            return {"model_response": "", "tool_calls": json.loads(tool_call)}
        elif self.fusion_method == "critic":
            num = 0
            while num < 3:
                try:
                    candidate_str = json.dumps(candidate_action)
                    # candidate_str = ""
                    # for idx, item in enumerate(candidate_action):
                    #     candidate_str += f"### Candidate plans {idx + 1}:\n"
                    #     for k, v in item.items():
                    #         candidate_str += f" - {k}: {str(v)}\n"

                    messages.append({"role": "user", "content": decide_tool_calling_prompt.format(candidate_plans=candidate_str)})
                    kwargs = {
                        "messages": messages,
                        "model": self.model_name.replace("-FC", "").replace("-Enhance", ""),
                        "temperature": self.temperature,
                        "store": False,
                    }
                    result = self.generate_with_backoff(**kwargs)[0] 
                    result = result.choices[0].message.content.strip() 
                    
                    match = re.search(r'```json(.*?)```', result, re.DOTALL)

                    if match:
                        fusion_result = match.group(1).strip()
                    else:
                        fusion_result = result
                    
                    fusion_result = json.loads(fusion_result)
                    
                    tool_calls = fusion_result.get("optimal_tool_call", {})
                    # processed_tool_calls = f"{tool_calls['name']}({','.join([f'{k}={repr(v)}' for k,v in tool_calls['parameters'].items()])})"
                    optimal_plan = fusion_result.get("optimal_plan", "")
                    if tool_calls["name"] == "response_to_user":
                        response = tool_calls["parameters"]["content"]
                        return {"model_response": response, "tool_calls": {}}
                    
                    return {"model_response": "", "tool_calls": tool_calls}
                except:
                    num += 1
                    print(f"Error in decision fusion and tool validator, retrying {num} time")
            # print(f"fusion failed, return the first plan: {candidate_plans[0]}")
            # return {"model_response": "", "tool_calls": candidate_plans[0]}
    

    @override
    def _query_FC(self, inference_data: dict):
        def to_chat_completion_response(api_response):
            from openai.types.chat import ChatCompletion, ChatCompletionMessage # , Choice, CompletionUsage
            
            """Convert tool-processed info into ChatCompletion-like structure with tool_calls"""

            choices = []
            for idx, item in enumerate(api_response.choices):

                tool_calls = None
                # 如果存在工具调用，转换结构
                if item.message.tool_calls:
                    tool_calls = []
                    for t_idx, t in enumerate(item.message.tool_calls[:1]):
                        # 统一 function.arguments 格式（确保字符串）
                        args = t.function.arguments
                        if not isinstance(args, str):
                            args = json.dumps(args, ensure_ascii=False)

                        tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id=t.id,
                                type="function",
                                function=dict(
                                    name=t.function.name,
                                    arguments=args
                                )
                            )
                        )

                # 构造 message
                msg = ChatCompletionMessage(
                    role=item.message.role,
                    content=item.message.content,
                    tool_calls=tool_calls,
                )

                # finish_reason 若有 tool call，一律设为 tool_calls
                finish_reason = item.finish_reason
                if tool_calls and finish_reason != "stop":
                    finish_reason = "tool_calls"

                choices.append(
                    Choice(
                        index=idx,
                        message=msg,
                        finish_reason=finish_reason or "stop",
                    )
                )

            usage = None
            if getattr(api_response, "usage", None):
                usage = CompletionUsage(
                    prompt_tokens=api_response.usage.prompt_tokens,
                    completion_tokens=api_response.usage.completion_tokens,
                    total_tokens=api_response.usage.total_tokens,
                )

            return ChatCompletion(
                id=str(uuid.uuid4()),
                object="chat.completion",
                created=int(time.time()),
                model=self.model_name.replace("-FC", "").replace("-Enhance", ""),
                choices=choices,
                usage=usage,
            )


        message = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        base_kwargs = {
            "messages": message,
            "model": self.model_name.replace("-FC", "").replace("-Enhance", ""),
            "temperature": self.temperature,
            "store": False,
            "tools": tools,
        }

        start_time = time.time()

        # === 多样性采样模式 ===
        if self.sample_num > 1:
            base_kwargs.update({
                "temperature": 1,
                "n": self.sample_num
            })

            response, _ = self.generate_with_backoff(**base_kwargs)
            # for item in response.choices:
            #     content = item.message.content
            #     try:
            #         tool_calls = [
            #             {func_call.function.name: func_call.function.arguments}
            #             for func_call in item.message.tool_calls
            #         ]
            #         tool_call_ids = [
            #             func_call.id for func_call in item.message.tool_calls
            #         ]
            #     except:
            #         tool_call_ids = []
                

            # candidate_tool_ids = [
            #     item.message.tool_calls[0].id 
            #     if item.message.tool_calls else None
            #     for item in response.choices
            # ]

            candidate_plans = [
                {"model_response": item.message.content, "tool_call": {"name": item.message.tool_calls[0].function.name, "parameters": json.loads(item.message.tool_calls[0].function.arguments)} if item.message.tool_calls else {}}
                for item in response.choices
            ]

            api_response = self.decision_fusion_and_tool_validator(
                candidate_plans, inference_data
            )
            api_response = self.format_chat_tool_response(api_response)

        # === 单一 prompt 模式 ===
        else:
            # print(f"message: {json.dumps(message, indent=2)}")
            api_response, _ = self.generate_with_backoff(**base_kwargs)
            api_response = to_chat_completion_response(api_response)
            # print(f"api_response: {api_response}")

        end_time = time.time()
        return api_response, end_time - start_time


    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        return super()._parse_query_response_FC(api_response)
        
    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        assert len(first_turn_message) == 1
        content = first_turn_message[0]["content"]
        content_question = copy.deepcopy(content) 
        pre_enhanced_content = (
                        "Before answering the user's question above, please first review the following related experiences:\n\n"
                    )
        involved_classes = inference_data["involved_classes"]
                    
        self.total_time += 1
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
        print(content)
        first_turn_message[0]["content"] = content
        inference_data["message"].extend(first_turn_message)
        return inference_data
    @override
    def _add_next_turn_user_message_FC(
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
        print(content)
        user_message[0]["content"] = content
        inference_data["message"].extend(user_message)
        return inference_data

    
    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        inference_data["involved_classes"] = test_entry["involved_classes"]
        return inference_data
        # return super()._pre_query_processing_FC(inference_data, test_entry)
        
    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        return {"message": [], "involved_classes": test_entry["involved_classes"], "tools": tools}

    # def _add_execution_results_FC(
    #     self,
    #     inference_data: dict,
    #     execution_results: list[str],
    #     model_response_data: dict,
    # ) -> dict:
    #     return super()._add_execution_results_FC(inference_data, execution_results, model_response_data)

    
        


    