import os
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def summarize_cluster_pattern(cluster_category, behaviors, available_tools, client, model=DEFAULT_MODEL):
    prompt = (
        f"The following is the tooling call path (in the form of a function call sequence) "
        f"for a user question belonging to category: {cluster_category}.\n\n"
        "Your task:\n"
        "1. Summarize the **common behavior patterns** of the following tooling call paths.\n"
        "2. The summary must be **comprehensive**:\n"
        "   - Identify mainstream steps.\n"
        "   - Describe potential **error-prone points** or pitfalls observed from the behaviors.\n"
        "   - **Add necessary steps to complete possible missing items** based on your reasoning.\n"
        "3. Provide a JSON-formatted generalized workflow that reflects this pattern.\n"
        "4. Replace all specific named entities with placeholders (e.g., <user_id_1>, <file_1>).\n\n"
        "Output Format:\n"
        "```json\n{\"summary\": \"\", \"step\": []}\n```\n\n"
        f"The available tools are {json.dumps(available_tools)}\n"
        "Now analyze the following tooling call paths:\n"
        + "\n".join(f"{idx+1}. {b}" for idx, b in enumerate(behaviors))
    )

    resp = {"summary": "", "step": []}
    for attempt in range(3):
        try:
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
            resp = {"summary": "", "step": None}

    return resp


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

            resp = summarize_cluster_pattern(cluster_category, behaviors, available_tools, client, model=model)
            embedding = embedding_model.encode([cluster_category])[0].tolist()
            intents[cls].append({
                "intent": cluster_category,
                "pattern": resp,
                "embedding": embedding,
            })

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
