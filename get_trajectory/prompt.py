prompt_template_reflection = """The following task, the model produced an incorrect answer to a given question. The task setting is that we provide relevant experiences (Retrieval Questions) to assist the model in answering. These experiences are presented in the form of a **Experience**, where the model retrieves past problems similar to the current question to improve accuracy. However, despite this support, the model still failed to generate the correct response. analyze the root causes based on the evaluation logs. Therefore, we need to find the reason for producing the error answer.

**Task Input:**

* **Question and Inference Trajectory:** {question_and_inference_trajectory}
* **Correct Response:** {correct_response}

Task: Produce a single, continuous explanation of **exactly 100 words**.

Guidelines:

* **Begin** with the direct cause of the error (this must be the first sentence).
* Then analyze the **underlying causes** beyond surface symptoms (for example: failing to check current state, omitting implicit actions, wrong call order, context loss across turns, parameter misuse, or overplanning).
* Finish with **detailed, targeted advice** that addresses the stated direct cause and underlying causes, and prescribes concrete mitigations for two causes.

Formatting rules: plain text only, no headings or code blocks, no extra commentary. Ensure the response is exactly 100 words and focused on actionable, cause-aligned recommendations.
The recommended output format is:
The problem is: … (Use the explicit tool calling trajectory to analyze the error) when asking this question, The direct cause of this problem is: … The root cause is: … To avoid this problem, when encountering similar cases in the future, one should …
"""


prompt_template_summary_1 = """In the following task, the model produced an incorrect answer to a given question. The task setting is that we provide relevant experiences (Retrieval Questions) to assist the model in answering. These experiences are presented in the form of a **Experience**, where the model retrieves past problems similar to the current question to improve accuracy. However, despite this support, the model still failed to generate the correct response.

Therefore, we need to analyze why the model doesn't give the right answer, and output modification advices accordingly.

**Task Input:**

* **Question and Inference Trajectory:** {question_and_inference_trajectory}
* **Correct Response:** {correct_response}

**Specific Requirements:**

1. Begin by analyzing why the model failed to produce the correct answer.
2. The output no more than 200 words.
3. The output must strictly follow this format: plain text only, no headings or code blocks, no extra commentary.

**Output writing guidelines:**

* The **first sentence** must clearly state the direct cause of the error.
* Then explain the **underlying causes** of current question, going beyond surface-level mistakes.
* Conclude with **specific and actionable advice** targeting at least two causes.

"""


prompt_template_summary_2 = """Therefore, based on the above analysis, we need to optimize the **Analysis & Advice** corresponding to the retrieval question by adding **guidance or hints** that encourage the model to reason more carefully and arrive at the correct answer for the new question and retaining most of the original Advice. For each retrieval question, there is a corresponding Analysis & Advice. You need to add new reflection experience to each Analysis & Advice by considering the reason why the retrieval question is not helpful for the new question. If the retrieval question is helpful for the new question, you can directly use the original Analysis & Advice. And Avoid major rewrites.

**Specific Requirements:**

2. Identify the weaknesses in the current Analysis & Advice.
3. Provide the optimized Analysis & Advice.
5. The output must strictly follow this JSON format:

```json
{output_format}
```

The output should be in ```json```.

**Reflection writing guidelines in optimized_advice:**

* The **first sentence** must clearly state the direct cause of the error.
* Then explain the **underlying causes** of current question, going beyond surface-level mistakes.
* Explain why the retrieved problems did not help the model obtain the correct answer.
* Conclude with **specific and actionable advice** targeting at least two causes.

**Reflection writing guidelines in new_advice:**

The new_advice should no more than 100 words.

* **Start with concrete, current-turn analysis.**
* **Then generalize into transferable rules, but include concrete details.** Note that **new_advice** should be **added after the original advice** rather than replacing it.
* The format is: When answering similar instruction, the model should…
"""


prompt_template_merge = """Below is the original reflection and several refined reflections created from different optimization perspectives.

**Task:** Summarize the optimization approaches reflected in these refined reflections, and then merge them into a single **new reflection** that comprehensively incorporates the different perspectives.

**Inputs:**

original reflection:
{original_reflection}

refined reflections:
{refined_reflections}

**Specific requirements:**

1. Analyze the refined reflections to identify the distinct optimization angles they apply.
2. Integrate these angles into one coherent and concise new reflection.
3. Ensure the new reflection preserves the strengths of the original while enhancing clarity, guidance, and reasoning support.
4. The output should be concise, comprehensive, and written in a natural language paragraph (not a list).

If the original reflection is empty, then you need to get a new reflection directly.

"""


prompt_template_merge_new = """Below is the original summary and several refined summary advices created from different optimization perspectives.

**Task:** Summarize the optimization approaches reflected in these refined summary advices, and then merge them into a single **new summary** that comprehensively incorporates the different perspectives.

**Inputs:**

original summary:
{original_summary}

refined summary advices:
{refined_summaries}

**Specific requirements:**

1. Analyze the refined summary advices to identify the distinct optimization angles they apply.
2. Integrate these angles into one coherent and concise new summary.
3. Ensure the new summary preserves the strengths of the original while enhancing clarity, guidance, and reasoning support.
4. The output should be concise, comprehensive, and written in a natural language paragraph (not a list).

"""


prompt_template_merge_reflection = """Below is the reflections and several advices created from different optimization perspectives.

**Task:** The reflections are created from different optimization perspectives. Please select the typical reflections by analyzing the optimization angles and output their id.

**Inputs:**

content:
{reflection}

**Specific requirements:**

The id is the position of each content, start from 1.

The output should be in a list of id. Each id should be separated by a comma. For example, [1,3,6,8].

"""


prompt_template_merge_new_2 = """Below is the summary and several advices created from different optimization perspectives.

**Task:** The summaries are created from different optimization perspectives. Please select the typical summaries by analyzing the optimization angles and output their id.

**Inputs:**

summary:
{summary}

**Specific requirements:**

The id is the position of each summary, start from 1.

The output should be in a list of id. Each id should be separated by a comma. For example, [1,3,6,8].

"""


MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

MULTI_TURN_FUNC_DOC_FILE_MAPPING_N = {v: k for k, v in MULTI_TURN_FUNC_DOC_FILE_MAPPING.items()}

OMIT_STATE_INFO_CLASSES = [
    "memory_kv.json",
    "memory_vector.json",
    "memory_rec_sum.json",
    "web_search.json",
]

GORILLA_TO_PYTHON = {
    "integer": "int",
    "number": "float",
    "float": "float",
    "string": "str",
    "boolean": "bool",
    "bool": "bool",
    "array": "list",
    "list": "list",
    "dict": "dict",
    "object": "dict",
    "tuple": "tuple",
    "any": "str",
    "byte": "int",
    "short": "int",
    "long": "int",
    "double": "float",
    "char": "str",
    "ArrayList": "list",
    "Array": "list",
    "HashMap": "dict",
    "Hashtable": "dict",
    "Queue": "list",
    "Stack": "list",
    "Any": "str",
    "String": "str",
    "Bigint": "int",
}

involved_classes_state = {
    "GorillaFileSystem": {
        "root": {
            "subdir1": {"file2.txt": {"type": "file", "content": "Python is fun"}},
            "subdir2": {"notes.md": {"type": "file", "content": "# Notes content"}},
            "file1.txt": {"type": "file", "content": "Hello World"},
        }
    },
    "MessageAPI": {
        "generated_ids": set(),
        "user_count": 4,
        "user_map": {
            "Alice": "USR001",
            "Bob": "USR002",
            "Catherine": "USR003",
            "Daniel": "USR004",
        },
        "inbox": [
            {"USR002": "My name is Alice. I want to connect."},
            {"USR003": "Could you upload the file?"},
            {"USR004": "Could you upload the file?"},
        ],
        "message_count": 3,
        "current_user": None,
    },
    "TwitterAPI": {
        "username": "john",
        "password": "john123",
        "authenticated": False,
        "tweets": {},
        "comments": {},
        "retweets": {},
        "following_list": ["alice", "bob"],
        "tweet_counter": 0,
    },
    "TicketAPI": {
        "ticket_queue": [],
        "ticket_counter": 1,
        "current_user": None,
    },
    "TradingBot": {
        "orders": {
            12345: {"id": 12345, "order_type": "Buy", "symbol": "AAPL", "price": 210.65, "amount": 10, "status": "Completed"},
            12446: {"id": 12446, "order_type": "Sell", "symbol": "GOOG", "price": 2840.56, "amount": 5, "status": "Pending"},
        },
        "account_info": {"account_id": 12345, "balance": 10000.0, "binding_card": 1974202140965533},
        "authenticated": False,
        "market_status": "Closed",
        "order_counter": 12446,
        "stocks": {
            "AAPL": {"price": 227.16, "percent_change": 0.17, "volume": 2.552, "MA(5)": 227.11, "MA(20)": 227.09},
            "GOOG": {"price": 2840.34, "percent_change": 0.24, "volume": 1.123, "MA(5)": 2835.67, "MA(20)": 2842.15},
            "TSLA": {"price": 667.92, "percent_change": -0.12, "volume": 1.654, "MA(5)": 671.15, "MA(20)": 668.20},
            "MSFT": {"price": 310.23, "percent_change": 0.09, "volume": 3.234, "MA(5)": 309.88, "MA(20)": 310.11},
            "NVDA": {"price": 220.34, "percent_change": 0.34, "volume": 1.234, "MA(5)": 220.45, "MA(20)": 220.67},
            "ALPH": {"price": 1320.45, "percent_change": -0.08, "volume": 1.567, "MA(5)": 1321.12, "MA(20)": 1325.78},
            "OMEG": {"price": 457.23, "percent_change": 0.12, "volume": 2.345, "MA(5)": 456.78, "MA(20)": 458.12},
            "QUAS": {"price": 725.89, "percent_change": -0.03, "volume": 1.789, "MA(5)": 726.45, "MA(20)": 728.00},
            "NEPT": {"price": 88.34, "percent_change": 0.19, "volume": 0.654, "MA(5)": 88.21, "MA(20)": 88.67},
            "SYNX": {"price": 345.67, "percent_change": 0.11, "volume": 2.112, "MA(5)": 345.34, "MA(20)": 346.12},
            "ZETA": {"price": 22.09, "percent_change": -0.05, "volume": 0.789, "MA(5)": 22.12, "MA(20)": 22.34},
        },
        "watch_list": ["NVDA"],
        "transaction_history": [],
        "random_seed": 1053520,
    },
    "TravelAPI": {
        "random_seed": 141053,
        "credit_card_list": {},
        "booking_record": {},
        "access_token": None,
        "token_type": None,
        "token_expires_in": None,
        "token_scope": None,
        "user_first_name": None,
        "user_last_name": None,
        "budget_limit": None,
    },
    "VehicleControlAPI": {
        "random_seed": 141053,
        "fuelLevel": 0.0,
        "batteryVoltage": 12.6,
        "engine_state": "stopped",
        "remainingUnlockedDoors": 4,
        "doorStatus": {"driver": "unlocked", "passenger": "unlocked", "rear_left": "unlocked", "rear_right": "unlocked"},
        "acTemperature": 25.0,
        "fanSpeed": 50,
        "acMode": "auto",
        "humidityLevel": 50.0,
        "headLightStatus": "off",
        "parkingBrakeStatus": "released",
        "_parkingBrakeForce": 0.0,
        "_slopeAngle": 0.0,
        "brakePedalStatus": "released",
        "brakePedalForce": 0.0,
        "distanceToNextVehicle": 50.0,
        "cruiseStatus": "inactive",
        "destination": "None",
        "frontLeftTirePressure": 32.0,
        "frontRightTirePressure": 32.0,
        "rearLeftTirePressure": 30.0,
        "rearRightTirePressure": 30.0,
    },
}
