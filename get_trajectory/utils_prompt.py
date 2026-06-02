import json


def _format_prompt(messages, function):
    """Format messages into Qwen chat template with tool call support."""
    formatted_prompt = ""

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

    if len(function) > 0:
        formatted_prompt += "<|im_start|>system\n"
        if messages[0]["role"] == "system":
            formatted_prompt += messages[0]["content"] + "\n\n"
        formatted_prompt += (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in function:
            formatted_prompt += f"\n{json.dumps(tool)}"
        formatted_prompt += (
            '\n</tools>\n\nFor each function call, return a json object with function name and arguments '
            'within <tool_call></tool_call> XML tags:\n<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
    else:
        if messages[0]["role"] == "system":
            formatted_prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"

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
                reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
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

            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                        formatted_prompt += "\n"
                    if "function" in tool_call:
                        tool_call = tool_call["function"]
                    formatted_prompt += '<tool_call>\n{"name": "'
                    formatted_prompt += tool_call["name"]
                    formatted_prompt += '", "arguments": '
                    if isinstance(tool_call["arguments"], str):
                        formatted_prompt += tool_call["arguments"]
                    else:
                        formatted_prompt += json.dumps(tool_call["arguments"])
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

    formatted_prompt += "<|im_start|>assistant\n"
    return formatted_prompt
