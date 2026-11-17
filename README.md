# chat-cli

> 一个命令行 LLM 聊天应用：流式输出、多轮对话、实时成本统计。
> W8 阶段项目，模型使用 DeepSeek `deepseek-flash`（OpenAI 兼容接口）。

## 功能

- [x] 非流式问答
- [x] 流式输出（逐字显示）
- [x] 多轮对话，`/clear` 清空、`/exit` 退出
- [x] `/cost` 显示累计 token 与金额
- [x] 429 / 超时自动重试，错误提示是中文
- [x] 单元测试不访问网络（14 个）

## 环境要求

- Python 3.11+
- DeepSeek API key

## 安装

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 配置

把 key 放进环境变量 `DEEPSEEK_API_KEY`，或者复制 `.env.example` 为 `.env` 再填写。

```bash
# 可选
export DEEPSEEK_MODEL=deepseek-flash
export DEEPSEEK_THINKING=0
```

## 运行

```bash
.venv/Scripts/python.exe chat.py
```

```
你> 你好
助手> 你好！有什么可以帮你的？
你> /cost
累计 输入 25 输出 12 tokens，约 ¥0.000049
```

## 项目结构

```
chat-cli/
├── config.py      # 环境变量与默认值
├── llm.py         # 调用、流式、重试、成本
├── chat.py        # 命令行入口
├── probe.py       # 打印真实响应结构，调试用
├── tests/         # 单元测试（不连网）
└── requirements.txt
```

## 设计说明

- **思考模式默认关闭**：V4 默认开思考，输出 token 会多十几倍，聊天场景不需要。
- **价格集中在 `llm.PRICE`**：变更只改一处，测试锁定计算逻辑。
- **测试不连网**：调用层和计算层分离，成本计算用构造的 `Usage` 测。

## 已知限制

- 多轮历史全量发送，长对话成本和延迟会平方级增长（W10 处理）
- 流式中途断网不会续写，直接报错
