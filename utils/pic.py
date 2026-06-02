from tqdm import tqdm
import os
import re
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json

def draw_pic(file_path):
    print(file_path)
    with open(file_path, "r") as f:
        output_result = json.load(f)
    analysis_result ={"Understanding Error": 0, "Visual Perception Error": 0, "Visual Reasoning Error": 0, "Textual Reasoning Error": 0, "Format Error": 0, "Other Error": 0}
    error_types = ["Visual Reasoning Error", "Visual Perception Error", "Understanding Error", "Textual Reasoning Error", "Format Error", "Other Error"]
    for item in output_result:
        for error in item["analysis"][1]:
            analysis_result[error] += 1
    # 归一化并过滤有效项（值大于0）
    cnt = len(output_result)
    normalized_result = {
        k: v / cnt for k, v in analysis_result.items() if v > 0
    }

    # 提取标签和数值
    labels = list(normalized_result.keys())
    sizes = list(normalized_result.values())

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, textprops={'fontsize': 12})
    plt.title(model_name, fontsize=16)
    plt.axis('equal')  # 使饼图为圆形

    # 保存图片
    plt.savefig(f"./pic/{model_name}.png")
    plt.close()

# if __name__ == "__main__":
output_path_o = "./MathVerse_result_analysis"
for dataset_name in ["MathVerse"]:
    for model_name in ["Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct", "gpt-4o", "Skywork-R1V2-38B", "QVQ-72B-Preview", "gemini-2.0-flash-thinking-exp"]:
        for infer_type in ["_direct", "_long_cot"]:
            if model_name == "Skywork-R1V2-38B":
                if infer_type == "_direct":
                    infer_type = "_mask"
            output_path = os.path.join(output_path_o, model_name+infer_type+".json")
            draw_pic(output_path)