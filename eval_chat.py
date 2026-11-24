"""跑 30 条评测集，按类别统计通过率。

用法：
    .venv/Scripts/python.exe eval_chat.py
    .venv/Scripts/python.exe eval_chat.py --verbose   # 打印失败原因和回答
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from context import ContextManager
from llm import ChatClient

SYSTEM = "你是一个简洁的中文助手，回答尽量短。"
CASES_PATH = Path(__file__).parent / "data" / "eval_cases.jsonl"
MAX_CONTEXT = 8000        # 评测不开压缩，固定上下文，保证结果可比


def load_cases(path: Path) -> list[dict]:
    """一行一个 JSON，跳过空行。"""
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def score(case: dict, answer: str) -> list[str]:
    """返回没通过的规则描述；空列表代表全过。"""
    failures = []

    # 统一转小写再比：Header / header / HEADER 都该算命中
    for group in case.get("must_include", []):
        if not any(word.lower() in answer.lower() for word in group):
            failures.append(f"缺少 {group[0]}")

    for word in case.get("must_not_include", []):
        if word.lower() in answer.lower():       # 只查当前这个词
            failures.append(f"包含 {word}")

    limit = case.get("max_chars")
    if limit and len(answer) > limit:
        failures.append(f"长度 {len(answer)} 超过 {limit} 字")

    return failures

def run_case(client: ChatClient, case: dict) -> tuple[str, int]:
    """跑一个用例，返回 (最后一条回答, 整个 case 消耗的输入 token)。"""
    ctx = ContextManager(SYSTEM, max_tokens=MAX_CONTEXT)
    answer = ""
    before = client.total_usage.prompt_tokens

    for turn in case["turns"]:
        ctx.add_user(turn)
        answer = client.chat(ctx.build_messages(), max_tokens=300)
        ctx.add_assistant(answer)

    return answer, client.total_usage.prompt_tokens - before


def print_table(results: list[dict]) -> None:
    by_cat: dict[str, list[int]] = {}
    for r in results:
        stat = by_cat.setdefault(r["category"], [0, 0])
        stat[1] += 1
        if not r["failures"]:
            stat[0] += 1

    print(f"{'类别':<14}{'通过':>8}{'通过率':>10}")
    for cat, (ok, total) in by_cat.items():
        print(f"{cat:<14}{ok:>4}/{total:<3}{ok / total * 100:>8.0f}%")

    ok = sum(1 for r in results if not r["failures"])
    total = len(results)
    print("-" * 32)
    print(f"{'总计':<14}{ok:>4}/{total:<3}{ok / total * 100:>8.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="打印失败原因和回答")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.disable(logging.INFO)

    client = ChatClient()
    cases = load_cases(CASES_PATH)
    results = []

    for case in cases:
        answer, tokens = run_case(client, case)
        failures = score(case, answer)
        results.append(
            {"id": case["id"], "category": case["category"],
             "failures": failures, "answer": answer, "tokens": tokens}
        )

        print(f"[{'PASS' if not failures else 'FAIL'}] #{case['id']:<2} {case['category']}")
        if args.verbose and failures:
            print(f"       原因：{'; '.join(failures)}")
            print(f"       回答：{answer.strip()[:120]}")

    print()
    print_table(results)
    print(f"\n总花费 {client.total_cost():.6f} 元")


if __name__ == "__main__":
    main()
