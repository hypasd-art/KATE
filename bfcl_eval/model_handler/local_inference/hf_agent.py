from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import os
import json
import requests
import re
from openai import OpenAI
import time
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Global Config: Initialize from environment variables or defaults
BASE_URL = os.getenv("BASE_URL", "http://210.75.240.154:29000/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-32B")
DEFAULT_MAX_WORKERS = 100
DEFAULT_MAX_OUTPUT_TOKENS = 3000


'''
cd Experience_Tool/
cd test/berkeley-function-call-leaderboard/
conda activate BFCL
export LLM_API_KEY=sk-WjE2sCHP0j67nAVb4aB0Af89Db294d7dBf0934494aB5EcCb
export BASE_URL=https://api.v3.cm/v1/
export LLM_MODEL=gpt-4o
python bfcl_eval/model_handler/api_inference/hf_multi_agent.py
'''

def top_k_similar_questions_reflection_and_summary(analysis_result, target_content, model=None, k=5, p = 0.6, involved_classes=None, skip_first_example=False):
    target_embeddings = model.encode([target_content])[0]
    information_results = []
    for involved_class in involved_classes:
        scores = []
        for question, item in analysis_result.items():
            if involved_class in item["involved_classes"]: # []
                question_embeddings = np.array(item["embedding"], dtype=np.float32) 
                score = util.cos_sim(target_embeddings, question_embeddings).item()
                scores.append((score, question))
        # 按分数降序排序，选前k个
        scores.sort(reverse=True)
        if skip_first_example:
            assert scores[0][1] == target_content
            scores = scores[1:]
        top_questions = [question for score, question in scores[:k] if score > p]
    information_results.extend(top_questions)
    return information_results

class TaskProcessingSystem:
    """
    Complete Task Processing System (Rewritten Version)
    Core Workflow: User Input → [Intent Recognizer + State Determiner] (Parallel) → [Multi-Subtask Planners] (Parallel) → 
                  Decision Fusion & Tool Validation → Tool Execution → Result Processing → Result Selection → Memory Update
    Features: Context reuse, parallel planners, API retry mechanism, structured output validation
    """

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS, args: Optional[Any] = None):
        """
        Initialize the system
        :param max_workers: Maximum threads for parallel execution pool
        :param args: External parameter object (must include `available_tools`, `max_output_tokens`, `print` attributes)
        """
        # Initialize memory module: Store interaction history, intents, and states
        self.api_key = args.api_key if args else LLM_API_KEY
        self.base_url = args.base_url if args else BASE_URL
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=BASE_URL)
        self.memory: Dict[str, List[Any]] = {
            "history": [],          # Full interaction history
            "history_intent": [],   # Historical intent recognition results
            "history_state": []     # Historical state determination results
        }
        self.current_intent: Optional[List[Dict]] = None  # Current intent (list for multi-intent support)
        self.current_state: Optional[Dict] = None         # Current system state
        self.fc_messages: List[Dict] = []                    # Conversation history (for LLM context)
        self.plan_messages: List[Dict] = []                    # Plan conversation history (for LLM context)
        self.model = args.model if args else LLM_MODEL
        
        # Tool configuration: Load from external args or use defaults
        self.tools: Dict[str, str] = args.tools
        
        # Thread pool & parameter setup
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self.max_output_tokens = args.max_output_tokens if args else DEFAULT_MAX_OUTPUT_TOKENS
        self.tool_process_threshold = args.tool_process_threshold if args else 1024  # Minimum tools required for result processing
        self.trajectory_retrieval = args.trajectory_retrieval if args else False
        self.skip_first_example = args.skip_first_example if args else False
        if self.trajectory_retrieval:
            self.retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.print_log = True # args.print if (args and hasattr(args, "print")) else False  # Log print toggle


    def _call_llm_api(self, messages: list[dict], tools: list[dict] = None, prompt: str = None):
        if prompt is not None:
            response = self.client.completions.create(
                model=self.model_name_huggingface,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            return response.model_dump()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        return response.model_dump()  
        

    def intent_recognizer(self, messages: List[Dict]) -> List[Dict]:
        history_user_messages = ""
        for m in messages:
            if m["role"] == "user":
                history_user_messages += m["content"] + "\n\n"
            elif m["role"] == "assistant":
                if "tool_calls" in m:
                    history_user_messages += "Tool Call from the Assistant: " + json.dumps(m["tool_call"]) + "\n" 
                else:
                    history_user_messages += "Assistant Response" + m["content"] + "\n\n"

        for m in messages:
            if m["role"] == "user":
                current_input = m["content"]
        # above 
        system_prompt = (
            f"The history of user questions and corresponding assistant responses is: \n{history_user_messages}\n"
            "The current question from the user is:\n"
            f"{current_input}\n\n"
            "Please summarize the user's intents or requests. \n"
            "Your output MUST strictly follow this JSON format (NO extra content):\n\n"
            "```json[\n"
            '  {"id": 1, "intent": "the first intent/request in one sentence"},\n'
            '  {"id": 2, "intent": "the second intent/request in one sentence"}\n'
            "]```\n"
            "- Always include at least one intent.\n"
            "- If multiple intents exist, split them clearly into separate entries.\n"
            "- Each intent must be a concise sentence (max 50 characters).\n"
            "- Focus primarily on the current question; previous questions are for reference only.\n"
            "- DO NOT include intents that were already fulfilled in earlier interactions."
            "The result must be a valid JSON array of objects with 'id' and 'intent' fields in ```json```."
        )

        # 3. Call LLM and parse structured results
        messages = [{"role": "user", "content": system_prompt}]
        for _ in range(3):  # Retry for structured output validation
            try:
                llm_result = self._call_llm_api(messages)
                llm_result = llm_result["choices"][0]["message"]["content"].strip()
                reasoning_content = ""
                if "</think>" in llm_result:
                    parts = llm_result.split("</think>")
                    reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    llm_result = parts[-1].lstrip("\n")
                # Extract JSON content between ```json and ```
                json_match = re.search(r"```json(.*?)```", llm_result, re.DOTALL)
                if json_match:
                    llm_result = json_match.group(1).strip()
                intent_data = json.loads(llm_result)
                # Validate result format
                if isinstance(intent_data, list) and len(intent_data) > 0:
                    # if self.print_log:
                        # print(f"Intent Recognition Result:\n{json.dumps(intent_data, indent=2)}")
                    self.memory["history_intent"].append(intent_data)
                    return intent_data
                raise ValueError("Intent result is not a non-empty list")
            except (json.JSONDecodeError, ValueError) as e:
                if self.print_log:
                    print(f"Intent Parsing Failed (Retrying): {str(e)} | LLM Raw Output: {llm_result}...")

        # Fallback handling (return default intent)
        default_intent = [{"id": 1, "intent": f"Process user request: {current_input}..."}]
        self.memory["history_intent"].append(default_intent)
        if self.print_log:
            print(f"Intent Recognition Fallback to Default: {default_intent}")
        return default_intent

    def state_determiner(self, messages: List[Dict]) -> Dict[str, Any]:
        history_user_messages = ""
        for m in messages:
            if m["role"] == "user":
                history_user_messages += m["content"] + "\n\n"
            elif m["role"] == "assistant":
                if "tool_calls" in m:
                    history_user_messages += "Tool Call from the Assistant: " + json.dumps(m["tool_call"]) + "\n" 
                else:
                    history_user_messages += "Assistant Response" + m["content"] + "\n\n"
        for m in messages:
            if m["role"] == "user":
                current_input = m["content"]
        # 2. Build prompt (fix original template syntax errors, clarify JSON format)
        # above
        system_prompt = (
            "You are a system state analyzer with tool-calling awareness. "
            "Your task is to determine the environmental state required to answer the user's question.\n\n"
            f"The interaction history is: \n{history_user_messages}\n"
            f"The current question from the user is:\n{current_input}\n\n"
            "Your output MUST strictly follow this JSON format (NO extra content):\n"
            "```json{\n"
            '  "analysis": "1-2 sentences analyzing the user\'s question and implicit state needs",\n'
            '  "state_requirements": [\n'
            '    {"id": 1, "state": "specific state to confirm (e.g., user login status, required context)"},\n'
            '    {"id": 2, "state": "another state if needed"}\n'
            '  ],\n'
            '  "required_tools_for_confirmation": ["tool1", "tool2"]  // Tools needed to verify states based on `state_requirements` (empty if none)\n'
            "}```\n"
            "- If no implicit states need confirmation, set `state_requirements` and `required_tools_for_confirmation` to empty lists.\n"
            "- Keep `analysis` concise and focused on state needs (not the answer itself)."
        )

        # 3. Call LLM and parse results
        messages = [{"role": "user", "content": system_prompt}]
        for _ in range(3):
            try:
                llm_result = self._call_llm_api(messages)
                llm_result = llm_result["choices"][0]["message"]["content"].strip()
                reasoning_content = ""
                if "</think>" in llm_result:
                    parts = llm_result.split("</think>")
                    reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    llm_result = parts[-1].lstrip("\n")
                # Extract JSON content between ```json and ```
                json_match = re.search(r"```json(.*?)```", llm_result, re.DOTALL)
                if json_match:
                    llm_result = json_match.group(1).strip()
                state_data = json.loads(llm_result)
                # Validate core fields
                required_fields = ["analysis", "state_requirements", "required_tools_for_confirmation"]
                if all(field in state_data for field in required_fields):
                    
                    self.memory["history_state"].append(state_data)
                    self.memory["history_state"].append(state_data)
                    return state_data
                raise ValueError("State result missing core fields")
            except (json.JSONDecodeError, ValueError) as e:
                if self.print_log:
                    print(f"State Parsing Failed (Retrying): {str(e)} | LLM Raw Output: {llm_result}...")

        # Fallback handling (default: no extra state requirements)
        default_state = {
            "analysis": f"User request: {current_input}... No additional state confirmation needed.",
            "state_requirements": [],
            "required_tools_for_confirmation": []
        }
        self.memory["history_state"].append(default_state)
        if self.print_log:
            print(f"State Determination Fallback to Default: {default_state}")
        return default_state

    def subtask_planner_planexec(self, planner_id: str = "PLANEXEC", current_intent: List[Dict[str, Any]] = None, current_state: Dict[str, Any] = None, tools: List = None, messages: List = None, plan_prompt: str = None) -> Dict[str, Any]:
        try:
            llm_response = self._call_llm_api(messages, tools=tools, prompt=plan_prompt)
            
            # Extract and clean LLM content (handle API response structure)
            if isinstance(llm_response, dict) and "choices" in llm_response:
                llm_content = llm_response["choices"][0]["message"]["content"].strip()
            else:
                llm_content = str(llm_response).strip()

            reasoning_content = ""
            if "</think>" in llm_content:
                parts = llm_content.split("</think>")
                reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                llm_content = parts[-1].lstrip("\n")

            # Remove any markdown formatting from LLM output
            cleaned_content = re.sub(r'```json|```', '', llm_content)
            plan_data = json.loads(cleaned_content)

            # 6. Validate tools_used matches actual tools in tasks

            # 7. Assemble final plan with validation results
            final_plan = {
                "strategy": "plan-execution-focused",
                "thought": plan_data["plan"],
                "tool_call": plan_data["next_execution"],
                "reasoning": reasoning_content,
            }
            return final_plan
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if self.print_log:
                print(f"[Plan Exec Planner {planner_id}] {error_msg}")
            return self._generate_fallback_plan(planner_id, "plan-execution-focused") 

    def fc_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        pass

    def subtask_planner_FC(self, planner_id: str = "Function-Calling", current_intent: List[Dict[str, Any]] = None, current_state: Dict[str, Any] = None, tools: List = None, messages: List = None) -> Dict[str, Any]:
        """
        Subtask planner using Function Calling (FC) strategy
        Generates plans optimized for tool function calls with parameter details
        """
        try:
            prompt = self.fc_prompt(messages, tools)
            result = self._call_llm_api(messages, tools, prompt=prompt) 
            llm_content = result["choices"][0]["text"] 
            tool_calls = self._extract_tool_calls(llm_content)
            
            # Build complete plan data
            plan_data = {
                "strategy": "function-calling",
                "thought": llm_content,  # Truncate long explanations
                "tool_call": tool_calls,
            }
            # }
            return plan_data
        except Exception as e:
            if self.print_log:
                print(f"[FC Planner {planner_id}] Error: {str(e)}")
            return self._generate_fallback_plan(planner_id, "function-calling")
    

    def decision_fusion_and_tool_validator(self, candidate_plans: List[Dict], current_intent: str, current_state: str, messages: List[Dict[str, str]], tools: List) -> Dict[str, Any]:
        """
        Fusion multiple candidate plans into an optimal plan and validate tools
        """
        
        try:
            attempt = 0
            while attempt < 3:
                try:
                    result = self._call_llm_api(messages)
                    result = result["choices"][0]["message"]["content"].strip()

                    reasoning_content = ""
                    if "</think>" in result:
                        parts = result.split("</think>")
                        reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                        result = parts[-1].lstrip("\n")
                    
                    match = re.search(r'```json(.*?)```', result, re.DOTALL)

                    if match:
                        fusion_result = match.group(1).strip()
                    else:
                        fusion_result = result

                    fusion_result = json.loads(fusion_result)
                    tool_calls = fusion_result.get("optimal_tool_call", {})
                    optimal_plan = fusion_result.get("optimal_plan", "")
                    return fusion_result["optimal_plan"], tool_calls
                except json.JSONDecodeError as e:
                    if self.print_log:
                        print(f"[Decision Fusion] JSON decode error: {str(e)}")
                    attempt += 1
                    if attempt == 3:
                        raise Exception
                    continue
                
            
        except Exception as e:
            if self.print_log:
                print(f"[Decision Fusion] Error: {str(e)}")
            return self._handle_fusion_failure(candidate_plans), {}

    def _extract_tool_calls(self, input_string):
        pass

    def tool_result_processor(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and normalize tool execution results
        """
        if len(tool_results) < self.tool_process_threshold:
            if self.print_log:
                print(f"[Result Processor] Not enough tool results to process ({len(tool_results)} < {self.tool_process_threshold})")
            return tool_results
        if self.print_log:
            print("\n[Result Processor] Analyzing tool outputs")

        system_prompt = f"""
You are a result processor. Analyze and normalize these tool execution results based on the conversation above:
{json.dumps(tool_results, indent=2)}

User intents to fulfill:
{json.dumps(self.current_intent, indent=2)}

Processing requirements:
1. Extract and summarize **key information** from all successful tool results.
2. Detect failures or partial successes, and suggest possible **workarounds or next steps**.
3. Reorganize and normalize results so they directly address the given user intents.
4. Validate the completeness of information, highlight **critical findings**, and identify any **gaps**.

Output strictly as a JSON object with:
- "processed_results": structured and organized key findings
- "additional_needs": list of missing information, unresolved gaps, or required follow-ups
        """

        messages = self.messages + [{"role": "system", "content": system_prompt}] #  + 

        try:
            result = self._call_llm_api(messages)
            processed = result
            
            if self.print_log:
                print(f"[Result Processor] Processed {len(tool_results)} tool outputs")
            return processed
            
        except json.JSONDecodeError as e:
            if self.print_log:
                print(f"[Result Processor] JSON decode error: {str(e)}")
            return self._generate_fallback_processing(tool_results)
        except Exception as e:
            if self.print_log:
                print(f"[Result Processor] Error: {str(e)}")
            return self._generate_fallback_processing(tool_results)

    def result_selector(self, processed_results: Dict[str, Any]) -> str:
        """
        Select and format the final response for the user
        """
        if self.print_log:
            print("\n[Result Selector] Generating final response")

        system_prompt = f"""
        You are a result selector. Create a clear, user-friendly response from:
        {json.dumps(processed_results, indent=2)}
        
        Follow these guidelines:
        1. Prioritize information that directly addresses user intents
        2. Explain failures transparently without technical jargon
        3. Structure response logically (use headings/bullets if helpful)
        4. Keep it concise but complete
        5. If information is missing, suggest next steps
        
        User intents reference:
        {json.dumps(self.current_intent, indent=2)}
        """

        messages = [{"role": "system", "content": system_prompt}] + self.messages
        return self._call_llm_api(messages)

    def trajectory_retrieve(self, content: str) -> List[Dict[str, Any]]:
        """
        Retrieve similar trajectories from memory
        """
        if not self.trajectory_result:
            return content
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
        return content # return top_k_similar_questions_reflection_and_summary(self.trajectory_result, content, 3, 0.5, self.involved_classes, self.skip_first_example)

    def memory_updater(self, result: Any) -> None:
        """
        Update memory with structured interaction data
        """
        interaction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "intent": self.current_intent,
            "state": self.current_state,
            "result_data": result,
            # "processing_duration": time.time() - self._process_start_time
        }

        # Maintain memory size (limit to last 100 interactions)
        self.memory["history"].append(interaction)
        if len(self.memory["history"]) > 100:
            self.memory["history"].pop(0)

        if self.print_log:
            print(f"\n[Memory Updater] Stored interaction (total: {len(self.memory['history'])})")

    # Helper methods
    def _generate_fallback_plan(self, planner_id: str, strategy: str) -> Dict[str, Any]:
        """Generate fallback plan when LLM plan generation fails"""
        return {
            "strategy": strategy,
            "thought": "",
            "tool_call": {},
        }

    def _handle_fusion_failure(self, candidate_plans: List[Dict]) -> Dict[str, Any]:
        """Handle decision fusion failure by selecting highest priority plan"""
        if not candidate_plans:
            return self._generate_fallback_plan("fusion_fallback", "emergency")
            
        # Select plan with highest priority
        highest_priority = next((plan for plan in candidate_plans if plan["strategy"] == "function-calling"), None)
        
        return highest_priority


# 使用示例
if __name__ == "__main__":
    # 创建系统实例
    system = TaskProcessingSystem()
    
    # 处理用户输入
    user_query = "请帮我查询今天的天气并计算未来三天的平均温度"
    user_question = "What's the weather?"
    result = system.process_input(user_question) # user_query
    
    # 输出结果
    print("\n最终结果:")
    print(result)
    
    # # 查看记忆内容（仅作演示）
    # print("\n记忆内容摘要:")
    # print(f"最近交互时间: {system.memory['last_state']['timestamp']}")
    # print(f"历史交互次数: {len(system.memory['history'])}")
