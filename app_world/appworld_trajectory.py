# Set your OPENAI Key
import os


os.environ["OPENAI_API_KEY"] = "..."
import copy
# This is a basic prompt template containing all the necessary onboarding information to solve AppWorld tasks. It explains the role of the agent and the supervisor, how to explore the API documentation, how to operate the interactive coding environment and call APIs via a simple task, and provides key instructions and disclaimers.

# You can adapt it as needed by your agent. You can also choose to bypass API docs app and build your own API retrieval, e.g., for FullCodeRefl, IPFunCall, etc, we asked an LLM to predict relevant APIs separately and put its documentation directly in the prompt.
PROMPT_TEMPLATE = """
USER:
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

For example, consider the task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

What is the password for my Spotify account?

ASSISTANT:
# Okay. Lets first find which apps are available to get the password by looking at the app descriptions.
print(apis.api_docs.show_app_descriptions())

USER:
[
  {
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  },
  {
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  },
  ...
  {
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  },
  {
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  },
  ...
]


ASSISTANT:
# Looks like the supervisor app could help me with that. Lets see what apis are available under this app.
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))


USER:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]


ASSISTANT:
# I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

USER:
{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {
    'success': [{'account_name': 'string', 'password': 'string'}],
    'failure': {'message': 'string'}
  }
}

ASSISTANT:
# Okay, it requires no arguments. So I can just call it directly.
print(apis.supervisor.show_account_passwords())

USER:
[
  {
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  },
  {
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  },
  ...
]


ASSISTANT:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)


USER:
dummy_spotify_pass

ASSISTANT:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)


USER:
Marked the active task complete.


----------------------------------------------

USER:
**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.

USER:
Using these APIs, now generate code to solve the actual task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

The code for reference:

# Okay. Lets first find which apps are available to get the password by looking at the app descriptions.
print(apis.api_docs.show_app_descriptions())

# Looks like the supervisor app could help me with that. Lets see what apis are available under this app.
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
{{code}}

You can not invode ``supervisor_profile = apis.supervisor.show_profile()`` in the code. And you need to call apis.api_docs.show_app_descriptions(), apis.api_docs.show_api_descriptions(app_name='<app>') and apis.api_docs.show_api_doc(app_name='<app>', api_name='<app_name>') before utilizing <app> and <app_name> others at first time.
You are not allowed to generate the entire code in one block. You must generate the code step by step.
Task:

{{ instruction }}
"""

from openai import OpenAI

base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
api_key = os.getenv("KEY")
client = OpenAI(api_key=api_key, base_url=base_url)
model = os.getenv("MODEL", "gpt-4o")


def call_llm(messages: list[dict]) -> str:
    """
    Call an LLM with a history of messages and return the response.
    """
    # Change this function to modify the base llm
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=400
    )
    text_response = ""
    if response.choices:
        text_response = response.choices[0].message.content
    return text_response

import re

from jinja2 import Template

from appworld.task import Task


class MinimalReactAgent:
    """A minimal ReAct Agent for AppWorld tasks."""

    def __init__(self, task, code):
        self.task = task
        self.code = code
        self.history: list[dict] = self.prompt_messages()
        

    def prompt_messages(self) -> list[dict]:
        """Builds prompt messages for the agent to solve self.task.instruction"""
        # Populate the fields of the prompt template with the task details
        dictionary = {"supervisor": self.task.supervisor, "instruction": self.task.instruction, "code": self.code}
        prompt = Template(PROMPT_TEMPLATE.lstrip()).render(dictionary)
        # Extract and return the OpenAI JSON formatted messages from the prompt
        messages: list[dict] = []
        last_start = 0
        for match in re.finditer("(USER|ASSISTANT|SYSTEM):\n", prompt):
            last_end = match.span()[0]
            if len(messages) == 0:
                if last_end != 0:
                    raise ValueError(
                        f"Start of the prompt has no assigned role: {prompt[:last_end]}"
                    )
            else:
                messages[-1]["content"] = prompt[last_start:last_end]
            mesg_type = match.group(1).lower()
            messages.append({"role": mesg_type, "content": None})
            last_start = match.span()[1]
        messages[-1]["content"] = prompt[last_start:]
        return messages

    def next_code_block(self, last_execution_output: str | None = None) -> str:
        """
        Asks Agent to generate next code block given last_execution_output and history.
        """
        # Add the last execution output as the user response to the history
        if last_execution_output is not None:
            if len(last_execution_output.strip()) > 5000:
                last_execution_output = last_execution_output[:5000] + "..." + "\nThe output is too long and we only keep the first 5000 characters."
            self.history.append({"role": "user", "content": last_execution_output})
        # Get the next code block based on the history.
        code = call_llm(self.history)
        # Add this code block to history as the assistant response
        self.history.append({"role": "assistant", "content": code})
        return code

    def get_history(self) -> list[dict]:
        """Returns the history of the agent."""
        return self.history
    
from concurrent.futures import ThreadPoolExecutor, as_completed
from appworld import AppWorld, load_task_ids
import json

# Split to evaluate on.
dataset_name = "train"  # Or dev, test_normal, test_challenge

# Experiment name. Experiment outputs are store in
# experiments/outputs/{experiment_name} relative to root ("." by default)
experiment_name = "minimal_react_agent"

# Max number of environment interactions per task
max_interactions = 50
if os.path.exists(f"./Experience/{experiment_name}_{dataset_name}.json"):
    with open(f"./Experience/{experiment_name}_{dataset_name}.json", "r") as f:
        saved_messages = json.load(f)
else:
    saved_messages = {}

saved_task_ids = list(saved_messages.keys())

messages = saved_messages # {}
# For each task in the dataset split
task_ids = load_task_ids(dataset_name)
for index, task_id in enumerate(task_ids):
    # Load the appworld environment for the task
    with AppWorld(
        task_id=task_id,
        ground_truth_mode=True,
        experiment_name=experiment_name,
    ) as world:
        if task_id in saved_task_ids:
            print(f"Task {task_id} skip")
            continue
        # Load the agent with the task to solve
        print("\n\n" + "*" * 20 + f" Task {index+1}/{len(task_ids)} ({task_id})  " + "*" * 20)
        print(world.task.instruction)
        path = world.task.ground_truth.compiled_solution_module().__file__
        # breakpoint()
        with open(path, "r") as f:
            code = f.read()
            # print(code)
        code = code.split("def solution(apis: ApiCollection, requester: Requester) -> None:")[-1]
        code_new = []
        for line in code.splitlines():
          # 1. 去掉一个缩进（如果有）
          if line.startswith("    "):
              line = line[4:]
          elif line.startswith("\t"):
              line = line[1:]
          # 2. 去掉行尾的空格
          line = line.rstrip()
          code_new.append(line)
        code = "\n".join(code_new)
        code_x = copy.deepcopy(code)
        # code = "\n".join(code_new)
        p = "The above code is the solution for current task, but some lines is not suitable and don't print the process and final result, you need to print the result for each result in step, you can refer to the above code when generating the code for dialogue."
        
        agent = MinimalReactAgent(world.task, code=code+"\n"+p)
        output: str | None = None
        # print(json.dumps(agent.get_history(), indent=2))
        # Until the task is completed or max_interactions is reached
        for _ in range(max_interactions):
            # ask the agent to generate the code block based on the history.
            code = agent.next_code_block(output)
            # execute the code in the world environment
            output = world.execute(code)
            # stop if agent has committed the task to be complete.
            if world.task_completed():
                message = agent.get_history()
                messages[task_id] = message
                with open(f"./Experience/{experiment_name}_{dataset_name}.json", "w") as f:
                    json.dump(messages, f, indent=2)
                break

def run_one_task(task_id, index, task_total, experiment_name, dataset_name, max_interactions):
    """执行单个 task，返回 message"""

    try:
        with AppWorld(
            task_id=task_id,
            ground_truth_mode=True,
            experiment_name=experiment_name,
        ) as world:

            print(f"\n\n{'*' * 20} Task {index+1}/{task_total} ({task_id}) {'*' * 20}")
            print(world.task.instruction)
            path = world.task.ground_truth.compiled_solution_module().__file__
            # breakpoint()
            with open(path, "r") as f:
                code = f.read()
                # print(code)
            code = code.split("def solution(apis: ApiCollection, requester: Requester) -> None:")[-1]
            code_new = []
            for line in code.splitlines():
              # 1. 去掉一个缩进（如果有）
              if line.startswith("    "):
                  line = line[4:]
              elif line.startswith("\t"):
                  line = line[1:]
              # 2. 去掉行尾的空格
              line = line.rstrip()
              code_new.append(line)
            code = "\n".join(code_new)
            p = "The above code is the solution for current task, but some lines is not suitable and don't print the process and final result, you can refer to the above code when generating the code for dialogue."
            

            agent = MinimalReactAgent(world.task, code=code+"\n"+p)
            output = None
            for _ in range(max_interactions):
                code = agent.next_code_block(output)
                print("\n\n" + "%" * 20 + " CODE " + "%" * 20 + "\n" + code)
                output = world.execute(code)
                print("\n\n" + "=" * 20 + " OUTPUT " + "=" * 20 + "\n" + output)
                
                if world.task_completed():
                    message = agent.get_history()
                    return message

            return None  # 超过 max_interactions 未完成

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}")
        return None


def run_tasks_multithread(task_ids, experiment_name, dataset_name, max_interactions=50, num_threads=1):
    messages = []
    task_total = len(task_ids)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交线程任务
        futures = {
            executor.submit(run_one_task, task_id, i, task_total,
                            experiment_name, dataset_name, max_interactions): task_id
            for i, task_id in enumerate(task_ids)
        }

        # 收集结果
        for future in as_completed(futures):
            task_id = futures[future]
            result = future.result()
            if result is not None:
                messages.append(result)

            # 保存增量结果（防止中途崩掉）
            with open(f"./Experience/{experiment_name}_{dataset_name}.json", "w") as f:
                json.dump(messages, f, indent=2)

    return messages

# run_tasks_multithread(load_task_ids(dataset_name), experiment_name, dataset_name)