
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
"""

reflection_prompt = """
Before invoking new tools, review the history of tool calls and their outcomes.

There are some suggestions for your reasoning:
1. Determine whether the previous tool calls were correct, sufficient, or complete. If a tool call failed or produced suboptimal results due to insufficient or missing parameters or functions, reflect on what information was lacking, how it could be inferred or obtained.
2. If issues exist (e.g., wrong parameters, missing calls, failed execution), explain briefly why they occurred.
3. Propose the correct next tool calls or improved reasoning steps needed to fix or continue the process.
"""

intent_prompt = """
Before invoking any tools, clearly identify the user's *current intent* based on the conversation history and the latest user message.

There are some suggestions for your reasoning:
1. Identify the user's current intent based on the conversation history and the latest user message.
2. Break down this intent into clear, actionable subtasks or goals.
3. Determine which tools (if any) are needed for each subtask, and specify their expected inputs and outputs.
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








react_prompt = """
You are a ReAct-style subtask planner.
Generate a plan with EXPLICIT REASONING and conditional steps for:
{current_intent}

Consider state requirements and potential uncertainties:
{current_state}

Available tools: {tools}

The conversation history:
{history_user_messages}

Your plan MUST include: Thought and Action

The output format is:
Thought: <reasoning>
Action: {{"name": <tool_name>, "parameters": <parameters>}}

You only need to Thought and give Action for next turn.

CORE REQUIREMENTS:
Each step MUST include:

1. "Thought" — a concise, focused explanation (one short paragraph) that explicitly states:
   - What information is needed or what specific problem this step addresses.
   - Why this step is necessary for fulfilling the user's intent.
   - How this step connects to prior and subsequent steps.
   - Any key uncertainties or assumptions the step depends on.

2. "Action" — a concrete, executable instruction containing:
   - "tool": the exact tool name from the provided tools list, or the string "finish" if no tool is required.
   - "parameters": an object with the exact parameters to pass to the tool (use `{{}}` if none).

ADDITIONAL RULES
- Use only the exact tool names supplied above; do not invent or assume additional tools.
- Do not include placeholders like `<value>` in the final text — provide concrete parameter values when possible.
- Provide only necessary steps; avoid redundant or speculative actions.
- Maintain clear sequential logic in your "thought" text; if steps are parallelizable, indicate that in the "thought".
- Keep each "thought" short and actionable (no more than one short paragraph).
- If no tool is required, use "finish" as the tool name.

        Return ONLY a valid text/plain response with these fields.
"""






plan_prompt = """
You are a **task planning and tool call assistant**.

Your role is to:
1. Decompose the user's request into actionable subtasks.
2. Plan the execution trajectory to achieve the user's goal.
3. Identify the next optimal tool call to execute, based on the current state. 

---

### Instructions

1. **Intent Decomposition**  
   - Break down the user request into discrete sub-intents or subtasks.  
   - Identify dependencies between subtasks.  

2. **Future Trajectory Planning**  
   - Using the decomposed intents, plan the remaining steps needed to fully satisfy the user request.  
   - Avoid repeating steps already executed in `executed_trace` and `conversation history`.  
   - Ensure sequential consistency and feasibility.  

3. **Next Tool Call Determination**  
   - From the planned trajectory, select **one next tool call** to execute based on the current state and conversation history.  
   - Ensure the tool exists in `available_tools` and its parameters are valid.  
   - Include any precondition checks if necessary (e.g., environment, authentication).  

---

### Output Format
Return a JSON object only, with the following structure in ```json```:

```json{
  "decomposed_intents": [
    {
      "subtask": "<sub-intent description>",
      "dependencies": ["<subtask_id>"]
    }
    ...
  ],
  "planned_trajectory": [
    {
      "step_id": 1,
      "description": "<planned action>",
      "tool_call": {
        "name": "<tool name>",
        "parameters": {}
      }
    }
    {
      "step_id": 2,
      "description": "<planned action>",
      "tool_call": {
        "name": "<tool name>",
        "parameters": {}
      }
    }
    ...
  ],
  "next_tool_call": {
    "analysis": "<analysis of the planned trajectory and the next action based on the current state and conversation history>",
    "name": "<tool name>",
    "parameters": {}
  }
}```
Avoid repeating steps already executed in `executed_trace` and `conversation history` in planned_trajectory.
The name of the tool should not contain 'functions' in its name and you do not need to add comments during the generation process.
if no tool call is needed, set next_tool_call={"name": "response_to_user", "parameters": {"content": "The response to the user"}}.
"""


