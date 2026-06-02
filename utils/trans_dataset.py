import json
import random
import os

import json
import os


results = []
with open("../bfcl_eval/data/BFCL_v4_memory.json", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            results.append(data)

part1 = []
part2 = []
for item in results:
    idx = int(item["id"].split("-")[-1])
    if idx % 2 == 0:
        part1.append(item)
    else:
        part2.append(item)

part1.sort(key=lambda x: int(x["id"].split("-")[-1]))
part2.sort(key=lambda x: int(x["id"].split("-")[-1]))

with open("../bfcl_eval/data/BFCL_v4_memory_training.json", "w") as f1:
    for item in part1:

        f1.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("../bfcl_eval/data/BFCL_v4_memory_testing.json", "w") as f2:
    for item in part2:

        f2.write(json.dumps(item, ensure_ascii=False) + "\n")

results = []
with open("../bfcl_eval/data/possible_answer/BFCL_v4_memory.json", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            results.append(data)


part1 = []
part2 = []
for item in results:
    idx = int(item["id"].split("-")[-1])
    if idx % 2 == 0:
        part1.append(item)
    else:
        part2.append(item)


part1.sort(key=lambda x: int(x["id"].split("-")[-1]))
part2.sort(key=lambda x: int(x["id"].split("-")[-1]))

with open("../bfcl_eval/data/possible_answer/BFCL_v4_memory_training.json", "w") as f1:
    for item in part1:
        # id_parts = item["id"].split("-")
        # item["id"] = "-".join(id_parts[:-1]) + "_training_" + id_parts[-1]
        f1.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("../bfcl_eval/data/possible_answer/BFCL_v4_memory_testing.json", "w") as f2:
    for item in part2:
        # id_parts = item["id"].split("-")
        # item["id"] = "-".join(id_parts[:-1]) + "_testing_" + id_parts[-1]
        f2.write(json.dumps(item, ensure_ascii=False) + "\n")