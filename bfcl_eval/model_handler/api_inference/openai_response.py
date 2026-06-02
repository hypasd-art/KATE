import json
import os
import time

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
from overrides import override
from openai import OpenAI, RateLimitError
from openai.types.responses import Response
from bfcl_eval.model_handler.utils import (
    top_k_similar_questions_reflection_and_summary,
    reasoning_enhance_prompt
)
import copy


class OpenAIResponsesHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OPENAI_RESPONSES
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.v3.cm/v1")

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        # OpenAI allows `system` role in the prompt, but it is meant for "messages added by OpenAI"
        # For our use case, it is recommended to use `developer` role instead.
        # See https://model-spec.openai.com/2025-04-11.html#definitions
        for prompt in prompts:
            if prompt["role"] == "system":
                prompt["role"] = "developer"

        return prompts

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
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result, has_tool_call_tag)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.responses.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }

        kwargs = {
            "input": message,
            "model": self.model_name.replace("-FC", ""),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }

        # OpenAI reasoning models don't support temperature parameter
        if "o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name:
            del kwargs["temperature"]

        # Non-reasoning models don't support reasoning parameter
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        inference_data["message"] = []

        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Response) -> dict:
        model_responses = []
        tool_call_ids = []

        for func_call in api_response.output:
            if func_call.type == "function_call":
                model_responses.append({func_call.name: func_call.arguments})
                tool_call_ids.append(func_call.call_id)

        if not model_responses:  # If there are no function calls
            model_responses = api_response.output_text

        # OpenAI reasoning models don't show full reasoning content in the api response,
        # but only a summary of the reasoning content.
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response.output,
            "tool_call_ids": tool_call_ids,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
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
        inference_data["message"].extend(
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
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": execution_result,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        kwargs = {
            "input": inference_data["message"],
            "model": self.model_name.replace("-FC", ""),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }

        # OpenAI reasoning models don't support temperature parameter
        if "o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name:
            del kwargs["temperature"]

        # Non-reasoning models don't support reasoning parameter
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Response) -> dict:
        # OpenAI reasoning models don't show full reasoning content in the api response,
        # but only a summary of the reasoning content.
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": api_response.output_text,
            "model_responses_message_for_chat_history": api_response.output,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
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
        inference_data["message"].extend(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data



class OpenAIResponsesEnhancedHandler(OpenAIResponsesHandler):
    def __init__(self, model_name, temperature, method, skip_first_example=False, information_dict=None, reasoning_enhance=False) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OPENAI_RESPONSES
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.v3.cm/v1")
        self.method = method.split(",")
        self.skip_first_example = skip_first_example
        self.reasoning_enhance=reasoning_enhance
        
        if "summary" in self.method:
            with open(information_dict, "r") as f:
                self.summary_result = json.load(f)
        else:
            self.summary_result = None
        if "reflection" in self.method:
            with open(information_dict, "r") as f:
                self.reflection_result = json.load(f)
        else:
            self.reflection_result = None

    def _add_next_turn_user_message_FC_reasoning_enhance(
        self, inference_data: dict, content: str
    ) -> dict:
        message: list[dict] = copy.deepcopy(inference_data["message"])
        tools = inference_data["tools"]
        message.append({"role": "user", "content": content + "\n\n" + reasoning_enhance_prompt})
        kwargs = {
            "input": message,
            "model": self.model_name.replace("-FC", "").replace("-Enhance", ""),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }

        # OpenAI reasoning models don't support temperature parameter
        if "o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name:
            del kwargs["temperature"]

        # Non-reasoning models don't support reasoning parameter
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        if len(tools) > 0:
            kwargs["tools"] = tools

        result, cost = self.generate_with_backoff(**kwargs)
        # model_response_data = self._parse_query_response_FC(result)
        model_response = result.output_text
        result = "There is some reasoning and suggestions to help you generating the correct tool-callings to solve the current turn quesiton:\n" + model_response
        # print(result)
        # breakpoint()
        return result

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }

        kwargs = {
            "input": message,
            "model": self.model_name.replace("-FC", "").replace("-Enhance", ""),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }

        # OpenAI reasoning models don't support temperature parameter
        if "o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name:
            del kwargs["temperature"]

        # Non-reasoning models don't support reasoning parameter
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _get_intent(
        self, inference_data: dict, content: str
    ) -> dict:
        message: list[dict] = copy.deepcopy(inference_data["message"])
        history_user_messages = [m["content"] for m in inference_data["message"] if m["role"] == "user"]
        history_user_messages = "\n\n".join(history_user_messages)
        
        
        message = [{"role": "user", "content": (
            "The history questions from the user are:\n" 
            + f"{history_user_messages}" 
            + "\n\nThe current question from the user is:\n" 
            + f"{content}" 
            + "\n\n" 
            + "Please summarize the user's intents or requests. \n"
            "Your output must strictly follow this JSON format:\n\n"
            "{\n"
            '  "intents": [\n'
            '    {"id": 1, "intent": "the first intent or request in one sentence"},\n'
            '    {"id": 2, "intent": "the second intent or request in one sentence"}\n'
            "  ]\n"
            "}\n\n"
            "- Always include at least one intent.\n"
            "- If multiple intents exist, split them clearly into separate entries.\n"
            "- Each intent should be a concise sentence.\n"
            "You should pay more attention to the intent of the user's current question, and previous questions are for reference only"
            )
        }]
        kwargs = {
            "input": message,
            "model": self.model_name.replace("-FC", "").replace("-Enhance", ""),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }

        # OpenAI reasoning models don't support temperature parameter
        if "o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name:
            del kwargs["temperature"]

        # Non-reasoning models don't support reasoning parameter
        else:
            del kwargs["reasoning"]
            del kwargs["include"]


        result, cost = self.generate_with_backoff(**kwargs)
        
        model_response = result_output_text
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            cleaned_response = parts[-1].lstrip("\n")
        else:
            cleaned_response = model_response
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # 如果模型没按要求输出 JSON，则兜底包装
            parsed = {"intents": [{"id": 1, "intent": cleaned_response}]}
        return parsed 

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # FC models use its own system prompt, so no need to add any message

        return {"message": [], "function": functions, "involved_classes": test_entry["involved_classes"]}


    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        assert len(first_turn_message) == 1
        content = first_turn_message[0]["content"]
        pre_enhanced_content = (
                        "\n\nBefore answering the user's question above, please first review the following related experiences:\n\n"
                    )
        involved_classes = inference_data["involved_classes"]
                    
        self.total_time += 1
        if self.summary_result is not None:
            num = 0
            intent = self._get_intent(inference_data, content)
            pattern = top_k_similar_questions_intent(self.intent_result, intent["intents"], 1, 0.5, involved_classes)
            if len(pattern) > 0:
                pre_enhanced_content += f"The user's intent is {json.dumps(intent['intents'])}\nThere are some behavior pattern for you to reference:\n"
                for item in pattern:
                    pre_enhanced_content += f"**Pattern Summary of {item['intent']}**: {item['pattern']['summary']}" # ["summary"]  \n**Step**: {json.dumps(item['step'])}\n
                    

            question = top_k_similar_questions_reflection_and_summary(self.summary_result, content, 3, 0.5, involved_classes,  self.skip_first_example)
            for idx, question_item in enumerate(reversed(question)):
                information = self.summary_result[question_item]
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
                    f"**Analysis & Advice:**\n{summary}\n\n"
                )
                
            if len(question) > 0:
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
                    + f"\nAttention the user question at current turn is: \n{content}"
                )
            print(self.retrieval_hit_time, self.total_time, self.retrieval_hit_time / self.total_time)
            
            '''print(f"Content: {content}")'''
            print(f"Content: {content}")
        first_turn_message[0]["content"] = content
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        assert len(user_message) == 1
        content = user_message[0]["content"]
        pre_enhanced_content = (
                        "\n\nBefore answering the user's question above, please first review the following related experiences:\n\n"
                    )
        tools = inference_data["function"]
        involved_classes = inference_data["involved_classes"]
        
        self.total_time += 1
        if self.summary_result is not None:
            num = 0
            intent = self._get_intent(inference_data, content)
            pattern = top_k_similar_questions_intent(self.intent_result, intent["intents"], 1, 0.5, involved_classes)
            if len(pattern) > 0:
                pre_enhanced_content += f"The user's intent is {json.dumps(intent['intents'])}\nThere are some behavior pattern for you to reference:\n"
                for item in pattern:
                    pre_enhanced_content += f"**Pattern Summary of {item['intent']}**: {item['pattern']['summary']} \n" # ["summary"] **Step**: {json.dumps(item['step'])}\n
                    # pre_enhanced_content += f"{json.dumps(item)}\n" # ["summary"]
                    
            question = top_k_similar_questions_reflection_and_summary(self.summary_result, content, 3, 0.5, involved_classes,  self.skip_first_example)
            for idx, question_item in enumerate(reversed(question)):
                information = self.summary_result[question_item]
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
                    f"**Analysis & Advice:**\n{summary}\n\n"
                )
                
            if len(question) > 0:
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
                    + f"\nAttention the user question at current turn is: \n{content}"
                )
            print(self.retrieval_hit_time, self.total_time, self.retrieval_hit_time / self.total_time)
            
            print(f"Content: {content}")
        user_message[0]["content"] = content
        inference_data["message"].extend(user_message)
        return inference_data


    
    # @override
    # def add_first_turn_message_FC(
    #     self, inference_data: dict, first_turn_message: list[dict]
    # ) -> dict:
    #     assert len(first_turn_message) == 1
    #     content = first_turn_message[0]["content"]
    #     pre_enhanced_content = (
    #                     "\n\nBefore answering the user's question above, please first review the following related experiences:\n\n"
    #                 )
                    
    #     if self.summary_result is not None:
    #         question = top_k_similar_questions_reflection_and_summary(self.summary_result, content, 5, 0.5, self.skip_first_example)
    #         for idx, question_item in enumerate(reversed(question)):
    #             information = self.summary_result[question_item]
    #             correct_response = information["answer"]
    #             summary = information["summary"]
    #             # Format the list of correct responses
    #             correct_response_str = "\n".join(
    #                 [f"- {resp}" for resp in correct_response]
    #             )
    #             if "all_reflection" in information:
    #                 if len(information["all_reflection"]) > 0:
    #                     pre_enhanced_content += (
    #                         f"### Example {idx + 1}\n"
    #                         f"**Question:** {question_item}\n\n"
    #                         f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
    #                         f"**Analysis & Advice:**\n{summary}\n\n"
    #                         f"**Error Cases:**\n{json.dumps(information['all_reflection'], indent=2)}\n\n"
    #                     )
    #                 else:
    #                     pre_enhanced_content += (
    #                     f"### Example {idx + 1}\n"
    #                     f"**Question:** {question_item}\n\n"
    #                     f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
    #                     f"**Analysis & Advice:**\n{summary}\n\n"
    #                 )
    #             else:
    #                 pre_enhanced_content += (
    #                     f"### Example {idx + 1}\n"
    #                     f"**Question:** {question_item}\n\n"
    #                     f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
    #                     f"**Analysis & Advice:**\n{summary}\n\n"
    #                 )
                
    #         if len(question) > 0:
    #             content += (
    #                 "\n\n"
    #                 + pre_enhanced_content
    #                 + "\n**Note**: You are not required to reference the information or examples above "
    #                 "if they are not directly relevant to the current user question. "
    #                 "Analyze the problem carefully, decide whether the retrieved information is useful, "
    #                 "and always apply reasoning before making any tool calls."
    #                 + f"\nAttention the user question at current turn is: {content}"
    #             )
            
        
    #     if self.reasoning_enhance:
    #         inference_reasoning = self._add_next_turn_user_message_FC_reasoning_enhance(inference_data, content)
    #         content += "\n\n" + inference_reasoning
    #     # print(f"Content: {content}")
    #     first_turn_message[0]["content"] = content
    #     inference_data["message"].extend(first_turn_message)
        
    #     return inference_data
    # @override
    # def _add_next_turn_user_message_FC(
    #     self, inference_data: dict, user_message: list[dict]
    # ) -> dict:
    #     assert len(user_message) == 1
    #     content = user_message[0]["content"]
    #     pre_enhanced_content = (
    #                     "\n\nBefore answering the user's question above, please first review the following related experiences:\n\n"
    #                 )
                    
    #     if self.summary_result is not None:
    #         question = top_k_similar_questions_reflection_and_summary(self.summary_result, content, 5, 0.5, self.skip_first_example)
    #         for idx, question_item in enumerate(reversed(question)):
    #             information = self.summary_result[question_item]
    #             correct_response = information["answer"]
    #             summary = information["summary"]
    #             # Format the list of correct responses
    #             correct_response_str = "\n".join(
    #                 [f"- {resp}" for resp in correct_response]
    #             )
    #             if "all_reflection" in information:
    #                 if len(information["all_reflection"]) > 0:
    #                     pre_enhanced_content += (
    #                         f"### Example {idx + 1}\n"
    #                         f"**Question:** {question_item}\n\n"
    #                         f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
    #                         f"**Analysis & Advice:**\n{summary}\n\n"
    #                         f"**Error Cases:**\n{json.dumps(information['all_reflection'], indent=2)}\n\n"
    #                     )
    #                 else:
    #                     pre_enhanced_content += (
    #                     f"### Example {idx + 1}\n"
    #                     f"**Question:** {question_item}\n\n"
    #                     f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
    #                     f"**Analysis & Advice:**\n{summary}\n\n"
    #                 )
    #             else:
    #                 pre_enhanced_content += (
    #                     f"### Example {idx + 1}\n"
    #                     f"**Question:** {question_item}\n\n"
    #                     f"**Correct Tool Calling Trajectory for Reference:**\n{correct_response_str}\n\n"
    #                     f"**Analysis & Advice:**\n{summary}\n\n"
    #                 )
                
    #         if len(question) > 0:
    #             content += (
    #                 "\n\n"
    #                 + pre_enhanced_content
    #                 + "\n**Note**: You are not required to reference the information or examples above "
    #                 "if they are not directly relevant to the current user question. "
    #                 "Analyze the problem carefully, decide whether the retrieved information is useful, "
    #                 "and always apply reasoning before making any tool calls."
    #                 + f"\nAttention the user question at current turn is: {content}"
    #             )
            
    #     if self.reasoning_enhance:
    #         inference_reasoning = self._add_next_turn_user_message_FC_reasoning_enhance(inference_data, content)
    #         content += "\n\n" + inference_reasoning
    #     # print(f"Content: {content}")
    #     user_message[0]["content"] = content
    #     inference_data["message"].extend(user_message)
        
    #     return inference_data

    

