# run.sh 使用文档

## 快速开始

```bash
# 使用默认配置 (Qwen3-32B, test_normal)
./run.sh

# 使用 GPT-4 + 检索增强 + 并行度4
./run.sh --gpt4 -r -p 4

# 运行两个数据集
./run.sh -d both

# 使用 Qwen3-8B
./run.sh --qwen8b
```

## 参数说明

### 数据集选项
- `-d, --dataset DATASET` - 数据集选择
  - `test_normal` (默认)
  - `test_challenge`
  - `both` - 运行两个数据集

### 模型预设
- `--gpt4` - GPT-4.1-2025-04-14
- `--qwen32b` - Qwen3-32B (默认)
- `--qwen8b` - Qwen3-8B

### 运行配置
- `-p, --parallel NUM` - 并行解码数量 (默认: 1)
- `-r, --retrieval` - 启用检索增强
- `--restart` - 重启实验

### 测试脚本
- 默认: `appworld_test.py`
- `--memp` - 使用 `memp.py`
- `--test2` - 使用 `appworld_test_2.py`

## 使用示例

### 基础用法
```bash
# 默认配置运行
./run.sh

# 指定数据集
./run.sh -d test_challenge

# 启用检索增强
./run.sh -r

# 设置并行度
./run.sh -p 4
```

### 组合使用
```bash
# GPT-4 + 检索 + 并行4 + 两个数据集
./run.sh --gpt4 -r -p 4 -d both

# Qwen3-8B + 检索 + 重启
./run.sh --qwen8b -r --restart

# 使用 memp.py + 检索
./run.sh --memp -r

# test_challenge + test2脚本 + 并行4
./run.sh --test2 -d test_challenge -p 4
```

## 实验命名规则

脚本会自动生成实验名称：

- **appworld_test.py**: `{dataset}_{model}_retrieval_{True/False}_{parallel}`
  - 例: `test_normal_Qwen3-32B_retrieval_True_4`

- **memp.py**: `memp_{dataset}_{model}`
  - 例: `memp_test_normal_Qwen3-32B`

- **appworld_test_2.py**: `m_{dataset}_{model}_retrieval_{True/False}_{parallel}`
  - 例: `m_test_challenge_gpt-4.1-2025-04-14_retrieval_True_4`

## 工作流程

每个实验执行以下步骤：

1. 设置环境变量 (BASE_URL, KEY, MODEL)
2. 生成实验名称
3. 运行 Python 测试脚本
4. 使用 `appworld evaluate` 评估结果

## 环境变量

脚本会自动设置以下环境变量：

```bash
export BASE_URL="..."  # API endpoint
export KEY="..."       # API key
export MODEL="..."     # 模型名称
```

## 常见组合

| 场景 | 命令 |
|------|------|
| 快速测试 | `./run.sh` |
| 完整评估 | `./run.sh -d both -r -p 4` |
| GPT-4基准 | `./run.sh --gpt4 -d both` |
| 对比测试 | `./run.sh --qwen32b -d both && ./run.sh --qwen8b -d both` |
| MEMP方法 | `./run.sh --memp -r -d both` |

## 注意事项

1. 确保对应的 Python 脚本存在
2. 确保 `appworld` 命令行工具已安装
3. API endpoint 需要可访问
4. 实验结果会自动保存到对应目录
