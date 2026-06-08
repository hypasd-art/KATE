# KATE App_World 项目文档 (中文版)

## 核心文件说明

### 1. Experience 目录
存储经验数据的目录，包含历史任务的执行轨迹和总结。

**文件列表：**
- `experience.json` - 原始经验数据，包含任务问题、执行轨迹、embedding向量和所需应用
- `experience_summary.json` - 经验总结数据，每个任务附带GPT生成的100字总结
- `minimal_react_agent_train.json` - 训练集任务的完整对话历史

### 2. appworld_test.py
**主测试脚本** - 使用ReAct模式在AppWorld环境中运行任务

**核心功能：**
- **MinimalReactAgent类**: 实现基于ReAct范式的智能体
  - 支持检索增强（从经验库中检索相似任务）
  - 支持并行解码（生成多个候选代码并聚合）
  - 逐步生成代码，每次一小段
  
**关键特性：**
- 检索增强 (`--retrieval_enhance`): 使用语义相似度从经验库检索top-k相似任务
- 并行解码 (`--parallel_decode`): 同时生成N个候选解决方案，去重后聚合
- 多进程执行: 使用ProcessPoolExecutor并行运行多个任务

**运行流程：**
1. 加载任务和经验数据
2. 为每个任务创建Agent实例
3. 迭代生成代码 → 执行 → 获取反馈
4. 直到任务完成或达到最大交互次数

### 3. appworld_trajectory.py
**轨迹生成脚本** - 基于ground truth代码生成对话轨迹

**核心功能：**
- 从AppWorld的ground truth获取标准解决方案代码
- 使用LLM将完整代码转换为逐步执行的对话形式
- 每步生成代码 + 打印中间结果

**作用：**
- 为训练集任务生成高质量的示例轨迹
- 这些轨迹存储到 `minimal_react_agent_train.json`
- 作为后续检索增强的经验库基础

### 4. get_trajectory.py
**经验提取脚本** - 从轨迹文件提取结构化经验

**核心功能：**
- 读取 `minimal_react_agent_train.json`
- 提取每个任务的：
  - 用户问题
  - 代码执行轨迹（分步骤）
  - 问题的embedding向量（用于检索）
  - 所需的API应用列表
- 输出到 `experience.json`

**数据结构：**
```json
{
  "问题文本": {
    "question": "...",
    "trajectory": "Code of Step 0:\n...\n---\nCode of Step 1:\n...",
    "embedding": [0.1, 0.2, ...],
    "required_apps": ["spotify", "supervisor"]
  }
}
```

### 5. get_trajectory_summary.py
**经验总结脚本** - 为每个经验生成抽象总结

**核心功能：**
- 使用GPT-4对每个任务的轨迹进行分析
- 生成100字的总结，包含：
  - 具体因果分析（如何解决问题）
  - 可迁移的通用规则（工具选择、参数处理、验证策略）
  - 从具体到抽象的推理模式

**并发处理：**
- 使用ThreadPoolExecutor (32线程)并行生成总结
- 带进度条显示
- 输出到 `experience_summary.json`

### 6. memp.py
**MEMP方法实现** - Memory-Enhanced Multi-step Planning

**与 appworld_test.py 的区别：**
- 不支持并行解码（单路径生成）
- 相同的检索增强机制
- 简化的Agent实现

**用途：**
- 作为baseline方法对比
- 单路径推理的参考实现

## 数据流程图

```
1. 生成轨迹
   appworld_trajectory.py
   ↓
   minimal_react_agent_train.json

2. 提取经验
   get_trajectory.py
   ↓
   experience.json

3. 生成总结
   get_trajectory_summary.py
   ↓
   experience_summary.json

4. 测试评估
   appworld_test.py / memp.py
   (使用 experience.json 进行检索增强)
   ↓
   experiments/outputs/{experiment_name}/
```

## 关键技术

### 检索增强 (Retrieval Enhancement)
1. 使用 `all-MiniLM-L6-v2` 模型对问题编码
2. 计算余弦相似度，筛选 score > 0.5 的top-3任务
3. 将相似任务的轨迹插入prompt作为参考

### 并行解码 (Parallel Decode)
1. 同时生成N个候选代码
2. 标准化并去重（去除注释和空白）
3. 如果所有候选相同 → 直接使用
4. 否则 → 调用LLM聚合选择最优代码

### 逐步生成 (Step-by-step)
- 每次只生成一小段代码
- 执行后获取反馈
- 根据反馈继续生成下一步
- 避免一次性生成完整解决方案

## 运行示例

```bash
# 使用检索增强 + 并行度4
./run.sh --gpt4 -r -p 4 -d test_normal

# 对应的Python命令
python appworld_test.py \
  --dataset_name test_normal \
  --experiment_name "test_normal_gpt-4.1_retrieval_True_4" \
  --retrieval_enhance \
  --parallel_decode 4 \
  --restart

# 使用MEMP方法
./run.sh --memp -r -d test_normal
```

## 配置要求

- Python 3.8+
- sentence-transformers
- openai
- appworld 包
- 模型文件：`all-MiniLM-L6-v2`
