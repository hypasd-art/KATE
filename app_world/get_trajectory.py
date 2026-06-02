import json
from tqdm import tqdm
import os
import re
import argparse
import json
import ast 
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from sentence_transformers import SentenceTransformer, util
import openai
from appworld import AppWorld, load_task_ids
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
MODEL = os.getenv("MODEL", "gpt-4o")

EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', '../models/all-MiniLM-L6-v2')
model = SentenceTransformer(EMBEDDING_MODEL_PATH)

with open("./Experience/minimal_react_agent_train.json") as f:
    data = json.load(f)
result = {}
trajectory = []
apps = []
for idx, data_item in data.items():
    with AppWorld(
        task_id=idx,
        ground_truth_mode=True,
        experiment_name="none",
    ) as world:
        print(idx)
        path = world.task.ground_truth.compiled_solution_module().__file__
        with open(path.replace("compiled_solution.py", "required_apps.json"), "r") as f:
            app = json.load(f)
        print(app)
        trajectory_item = ""
        num = 0
        for message_item in data_item[15:]:
            # breakpoint()
            if message_item["role"] == "assistant":
                trajectory_item += f"Code of Step {num}:\n" + message_item["content"] + "\n\n---\n\n"
                num += 1
        trajectory.append(trajectory_item)
        user_question = data_item[14]["content"].split("Task:\n\n")[-1].strip()
        result[user_question] = {
            "question": user_question,
            "trajectory": trajectory_item,
            "embedding": model.encode(user_question).tolist(),
            "required_apps": world.task.ground_truth.required_apps,
        }
        apps.append(app[0])
apps = list(set(apps))
print(apps)
with open("./Experience/experience.json", "w") as f:
    json.dump(result, f, indent=4)