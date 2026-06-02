intent_state_prompt = """Generate a plan with EXPLICIT REASONING and conditional steps for:
{current_intent}

Consider state requirements and potential uncertainties:
{current_state}"""

state_prompt = """
Before invoking any tools, carefully identify the *environment* and the *state* required to answer the user's question, because these may influence both the tool selection and the parameters for tool calls.

There are some suggestions for your reasoning:
1. Analyze the user's question to detect any implicit state dependencies (e.g., user login status, file existence, context variables).
2. Determine what specific states must be confirmed before continuing.
3. If verification is required, decide which tools should be invoked to confirm those states. If no state verification is needed, proceed with reasoning toward tool selection or response generation.

You do not have to fully adhere to the above suggestions. But you need to analyze the relevant points in the conversation history about the state requirements in the thinking process.
"""

reflection_prompt = """
Before invoking new tools, review the history of tool calls and their outcomes.

There are some suggestions for your reasoning:
1. Determine whether the previous tool calls were correct, sufficient, or complete. If a tool call failed or produced suboptimal results due to insufficient or missing parameters or functions, reflect on what information was lacking, how it could be inferred or obtained.
2. If issues exist (e.g., wrong parameters, missing calls, failed execution), explain briefly why they occurred.
3. Analyze future multi-step tool calls during the analysis process, rather than just focusing on the next step.

You do not have to fully adhere to the above suggestions. But you need to analyze the relevant points in the conversation history about the correctness and necessary of previous tool call in the thinking process.
"""

intent_prompt = """
Before invoking any tools, clearly identify the user's *current intent* based on the conversation history and the latest user message.

There are some suggestions for your reasoning:
1. Identify the user's current intent based on the conversation history and the latest user message.
2. Break down this intent into clear, actionable subtasks or goals.
3. Determine which tools (if any) are needed for each subtask, and specify their expected inputs and outputs. Your reasoning should focus on *clarity* (what the user wants), *structure* (how to achieve it), and *efficiency* (which tool or reasoning step should come next).

You do not have to fully adhere to the above suggestions. But you need to analyze the relevant points in the conversation history about the intent requirements in the thinking process.
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
Only one tool call is allowed in the optimal_tool_call.
If no tool call is needed, set "optimal_tool_call": {{"name": "response_to_user", "parameters": {{"content": "The response to the user"}}}}.
"""


decide_tool_calling_prompt_one_turn = """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- History Conversation:
{history_messages}

- Available Tools:
{available_tools}

- Candidate Plans:
{candidate_plans}  

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
"""


decide_prompt_dict = {
    "normal": decide_tool_calling_prompt,
    "function_and_parameters_check": """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- Candidate tool calls: {candidate_plans} 

TASK RULES FOR FUNCTION AND PARAMETERS EVALUATION:
1) Ensure the tool name exists in available_tools.
2) Ensure all required parameters are present.
3) Ensure argument value types match required types, or can be safely coerced:
   - numbers → int/float as needed
   - strings → normalized (e.g., dates → ISO8601, trimmed whitespace)
   - enums → verify allowed values; case-normalize if safe
4) Reject any extra/unknown parameters not in schema.
5) If validation fails, produce a user-facing response asking for missing or ambiguous information.
6) Otherwise, return exactly one normalized valid tool call.

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
""",
    "logic_check": """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- Candidate tool calls: {candidate_plans}

TASK RULES FOR LOGIC EVALUATION:
1) Is the candidate action a reasonable next step toward satisfying the user's intent?
2) Does it avoid redundant operations or unnecessary tool calls?
3) Does it avoid hallucinating capabilities the tools do not have?
4) If the user intent is ambiguous, return a clarification question instead of making assumptions.
5) Ensure tool call sequences follow logical order (e.g., must login before reading private data).
6) If multiple candidates are present, choose the one that best satisfies the user's intent with minimal assumptions.

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
""", 
    "current_state_check": """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- Candidate tool calls: {candidate_plans}

TASK RULES FOR CURRENT STATE EVALUATION:
1) Are all prerequisites satisfied?
   - authentication tokens present?
   - prior selections or confirmations completed?
   - resource identifiers available in state?
2) If a tool call depends on missing session_state information, ask the user to provide it.
3) If the previous tool calls have already completed part of the task, avoid repeating them.
4) If the system is already in the correct state, proceed with the next logical step.
5) If the tool call contradicts known state (invalid resource id, expired token, wrong sequence), produce a correction or a user-facing clarification.


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
"""
}


decide_prompt_dict_one_turn = {
    "normal": decide_tool_calling_prompt_one_turn,
    "function_and_parameters_check": """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- History Conversation:
{history_messages}

- Available Tools:
{available_tools}

- Candidate Plans:
{candidate_plans} 

TASK RULES FOR FUNCTION AND PARAMETERS EVALUATION:
1) Ensure the tool name exists in available_tools.
2) Ensure all required parameters are present.
3) Ensure argument value types match required types, or can be safely coerced:
   - numbers → int/float as needed
   - strings → normalized (e.g., dates → ISO8601, trimmed whitespace)
   - enums → verify allowed values; case-normalize if safe
4) Reject any extra/unknown parameters not in schema.
5) If validation fails, produce a user-facing response asking for missing or ambiguous information.
6) Otherwise, return exactly one normalized valid tool call.

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
""",
    "logic_check": """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- History Conversation:
{history_messages}

- Available Tools:
{available_tools}

- Candidate Plans:
{candidate_plans}   

TASK RULES FOR LOGIC EVALUATION:
1) Is the candidate action a reasonable next step toward satisfying the user's intent?
2) Does it avoid redundant operations or unnecessary tool calls?
3) Does it avoid hallucinating capabilities the tools do not have?
4) If the user intent is ambiguous, return a clarification question instead of making assumptions.
5) Ensure tool call sequences follow logical order (e.g., must login before reading private data).
6) If multiple candidates are present, choose the one that best satisfies the user's intent with minimal assumptions.

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
""", 
    "current_state_check": """
You are a tool calling agent. Based on the conversation history, available tools, and candidate tool calls provided.
Your task is to evaluate multiple candidate tool calls generated for the user's questions and assistant responses, analyze their correctness, and produce a single **optimal plan** along with a **validated tool call**.

---

### Inputs
- History Conversation:
{history_messages}

- Available Tools:
{available_tools}

- Candidate Plans:
{candidate_plans}  

TASK RULES FOR CURRENT STATE EVALUATION:
1) Are all prerequisites satisfied?
   - authentication tokens present?
   - prior selections or confirmations completed?
   - resource identifiers available in state?
2) If a tool call depends on missing session_state information, ask the user to provide it.
3) If the previous tool calls have already completed part of the task, avoid repeating them.
4) If the system is already in the correct state, proceed with the next logical step.
5) If the tool call contradicts known state (invalid resource id, expired token, wrong sequence), produce a correction or a user-facing clarification.

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
"""
}