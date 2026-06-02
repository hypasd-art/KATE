import os
import json
import math
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from prompt import MULTI_TURN_FUNC_DOC_FILE_MAPPING

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "/home/yphao/Experience_Tool/test/gorilla/berkeley-function-call-leaderboard/all-MiniLM-L6-v2"
)
FUNC_DOC_DIR = os.getenv("FUNC_DOC_DIR", "../bfcl_eval/data/multi_turn_func_doc")


def cluster_questions(info_list, client, model=DEFAULT_MODEL, min_clusters=2, max_clusters=15):
    questions = [item["question"] for item in info_list]
    embeddings = np.array([item["embedding"] for item in info_list])
    print(f"Clustering {len(questions)} questions...")

    best_score, best_k, best_labels = -1, None, None
    for k in range(min_clusters, min(max_clusters, len(questions))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels

    print(f"Best k={best_k}, silhouette={best_score:.4f}")

    clusters = {}
    for q, label in zip(questions, best_labels):
        clusters.setdefault(label, []).append(q)

    cluster_labels = {}
    for cluster_id, cluster_qs in clusters.items():
        prompt = (
            "You are an intent clustering assistant. Below is a set of user questions. "
            "Please summarize their common intent.\n"
            "Output a short sentence (no more than 20 words) to describe the common intent:\n\n"
            + "\n".join(f"- {q}" for q in cluster_qs[:50])
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        cluster_labels[cluster_id] = resp.choices[0].message.content.strip()

    for q, label in zip(questions, best_labels):
        for index, item in enumerate(info_list):
            if item["question"] == q:
                info_list[index]["cluster_id"] = int(label)
                info_list[index]["cluster_label"] = cluster_labels[label]

    return info_list, clusters, cluster_labels


def process_single_cluster(
    cluster_id: str,
    questions: List[str],
    info_dict: Dict,
    client,
    model: str = DEFAULT_MODEL,
    batch_size: int = 10,
) -> Tuple[str, str, Dict]:
    behaviors = [
        f"Question:{q}\nTooling Call Path:{json.dumps(info_dict[q]['steps'])}"
        for q in questions
        if "steps" in info_dict[q]
    ]
    if not behaviors:
        return cluster_id, "", {}

    cluster_category = info_dict[questions[0]]["cluster_label"]
    num_batches = math.ceil(len(behaviors) / batch_size)
    batch_summaries = []

    for i in range(num_batches):
        batch = behaviors[i * batch_size: (i + 1) * batch_size]
        prompt = (
            f"The following is the tooling call path for questions in category: {cluster_category}.\n"
            "Summarize the common behavior pattern and output concise JSON steps.\n"
            "Be comprehensive — cover edge cases, not just the mainstream flow.\n\n"
            "Output Format: [{\"pattern_1\": {\"desc\": \"\", \"step\": []}}, ...]\n\n"
            + "\n".join(f"{idx+1}. {b}" for idx, b in enumerate(batch))
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        batch_summaries.append(resp.choices[0].message.content.strip())

    if len(batch_summaries) > 1:
        merge_prompt = (
            f"The following are behavior pattern summaries for category: {cluster_category}.\n"
            "Consolidate them into one comprehensive pattern.\n"
            "Output Format (single pattern): {\"desc\": \"\", \"step\": []}\n\n"
            + "\n".join(f"- {s}" for s in batch_summaries)
        )
        final_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": merge_prompt}],
        )
        final_pattern = final_resp.choices[0].message.content.strip()
    else:
        final_pattern = batch_summaries[0] if batch_summaries else ""

    updated_info = {q: {"behavior_pattern": final_pattern} for q in questions}
    return cluster_id, final_pattern, updated_info


def summarize_cluster_behaviors_thr(
    info_dict: Dict,
    clusters: Dict,
    client,
    model: str = DEFAULT_MODEL,
    batch_size: int = 30,
    max_workers: int = 16,
) -> Tuple[Dict, Dict]:
    cluster_patterns = {}
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_cluster, cluster_id, questions, info_dict, client, model, batch_size)
            for cluster_id, questions in clusters.items()
        ]
        for future in as_completed(futures):
            cluster_id, final_pattern, updated_info = future.result()
            if final_pattern:
                cluster_patterns[cluster_id] = final_pattern
                for q, updates in updated_info.items():
                    info_dict[q].update(updates)
                with print_lock:
                    print(f"Cluster {cluster_id} processed.")

    return info_dict, cluster_patterns


def run(input_file, output_file, func_doc_dir=FUNC_DOC_DIR, model=DEFAULT_MODEL):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    with open(input_file, "r") as f:
        info_dict = json.load(f)

    infor_new = {k: [] for k in MULTI_TURN_FUNC_DOC_FILE_MAPPING.keys()}
    intents = {k: [] for k in MULTI_TURN_FUNC_DOC_FILE_MAPPING.keys()}

    for item in info_dict.values():
        for cls in item["involved_classes"]:
            if cls in infor_new:
                infor_new[cls].append(item)

    for cls, items in infor_new.items():
        if not items:
            continue
        items, clusters, cluster_labels = cluster_questions(items, client, model=model)

        available_tools = []
        doc_file = os.path.join(func_doc_dir, MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls])
        with open(doc_file, "r") as f:
            for line in f:
                available_tools.append(json.loads(line))

        for cluster_id, cluster_qs in clusters.items():
            cluster_category = cluster_labels[cluster_id]
            behaviors = []
            for q in cluster_qs:
                for item in items:
                    if item["question"] == q:
                        behaviors.append(
                            f"Question:{q}\nTooling Call Path:{json.dumps(item['answer'])}"
                        )

            resp = {"desc": "", "step": None}
            for attempt in range(3):
                try:
                    prompt = (
                        f"The following is the tooling call path for questions in category: {cluster_category}.\n"
                        "Summarize the common behavior pattern and output concise JSON steps.\n"
                        "Output Format: ```json{\"desc\": \"\", \"step\": []}```\n\n"
                        + "\n".join(f"{idx+1}. {b}" for idx, b in enumerate(behaviors))
                    )
                    raw = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = raw.choices[0].message.content.strip()
                    match = re.search(r"```json(.*?)```", text, re.DOTALL)
                    if match:
                        text = match.group(1).strip()
                    resp = json.loads(text)
                    assert isinstance(resp, dict)
                    break
                except Exception:
                    resp = {"desc": "", "step": None}

            embedding = embedding_model.encode([cluster_category])[0].tolist()
            try:
                intents[cls].append({
                    "intent": cluster_category,
                    "pattern": resp.get("step"),
                    "embedding": embedding,
                })
            except Exception:
                print(f"Failed to append intent for cluster {cluster_id}: {resp}")

    with open(output_file, "w") as f:
        json.dump(intents, f, indent=4)

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input_file", type=str, default="../Experience/BFCL_v4_multi_turn_base_training_summary_with_embedding.json")
    parser.add_argument("--output_file", type=str, default="../Experience/intent.json")
    parser.add_argument("--func_doc_dir", type=str, default=FUNC_DOC_DIR)
    args = parser.parse_args()

    run(args.input_file, args.output_file, func_doc_dir=args.func_doc_dir, model=args.model)
