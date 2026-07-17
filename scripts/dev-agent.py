#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dev-agent.py — 项目推进流水线「目标仓自治 dev agent」（ADR-0003），Python 仓版。

对标 cc-web-control/scripts/dev-agent.mjs（Node 版），契约一一对应：
  CLI（--prd/--source/--base/--branch-prefix/--dry-run）、退出码（0/10/11/12/13/99）、
  stdout 单行 JSON（字段名与 mjs 完全一致 → pa-dispatch 解析端零改动）、SPEC #27 stall 刹车。

Python / 本仓适配（与 mjs 的差异）：
  - agent loop 走 Python `claude-agent-sdk` 的 `query()`（async 生成器），options 用 snake_case
    （permission_mode/setting_sources/allowed_tools/max_turns/max_budget_usd —— 与 Node SDK 同源）。
  - 验证闸是 `python tests/run_tests.py`（unittest，非 pytest、非 npm test）；stall 刹车按此识别。
  - conda env：dev-agent.py 由 ashare-llm-analyst env 的 python 启动（sys.executable 即 env python），
    把其目录前置进 SDK 的 PATH → agent 的裸 `python` 命令自动落在该 env（baostock/akshare/sklearn 可用）。
  - 默认基点 base=master（本仓默认分支；branch protection 下永不直推主干）。
  - 新工作落 src/（版本化核心）；out//back//deep-search/ 仅参考。

用法（pa-dispatch 自动调，或人手动）：
  python scripts/dev-agent.py --prd <prd.md> [--source <signal.md>] [--base master] [--dry-run]

退出码：0=成功（开 PR 或 dry-run 完成）| 10=PRD 缺失/读不到 | 11=SDK dev loop 失败 |
        12=stalled（SPEC #27 主动刹车）| 13=git/push/PR 失败 | 99=未捕获

stdout 只输出最终一行 JSON（供 pa-dispatch 解析）；过程日志全走 stderr。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

REPO_ROOT = Path.cwd()

# ─── 仓内主动刹车 + 监控常量（SPEC §决策#27，对齐 mjs）────────────────
WRITE_TOOLS = {"Edit", "Write", "MultiEdit"}   # 写类工具（Bash 跑 test/git，不计入"尝试修代码"）
N_STALL = 3                  # verifiedRed 后连续 N 轮无写类 tool_use → stalled
INPUT_TRUNC = 500            # tool_use.input 落盘截断
MAX_BUDGET = 10              # maxBudgetUsd（降级兜底，宽松）
STATE_RUNS_DIR = REPO_ROOT / "state" / "runs"


def parse_args(argv: list[str]) -> dict:
    out = {"prd": None, "source": None, "base": "master", "dry_run": False,
           "branch_prefix": "pa-dev", "help": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--prd": out["prd"] = argv[i + 1]; i += 1
        elif a == "--source": out["source"] = argv[i + 1]; i += 1
        elif a == "--base": out["base"] = argv[i + 1]; i += 1
        elif a == "--branch-prefix": out["branch_prefix"] = argv[i + 1]; i += 1
        elif a == "--dry-run": out["dry_run"] = True
        elif a in ("-h", "--help"): out["help"] = True
        i += 1
    return out


HELP = """dev-agent.py — 目标仓自治 dev agent（ADR-0003，Python 仓版）
用法: python scripts/dev-agent.py --prd <prd.md> [--source <signal.md>] [--base master] [--dry-run]
  --prd            PRD 文件路径（必填，来自控制面 pa-prd）
  --source         触发该任务的信号文件（可选，附给 dev agent 做上下文）
  --base           分支基点（默认 master；branch protection 下永不直推主干）
  --dry-run        只跑到"改完代码 + 本地 test"，不 commit/push/开 PR
退出码: 0 OK | 10 无/读不到 PRD | 11 SDK 失败 | 12 stalled | 13 git/PR 失败 | 99 未捕获"""


def git(args: list[str]) -> str:
    """跑 git（arg 数组，无 shell 注入）；非零抛异常，带 stderr。"""
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=120)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or "").strip() or f"git {args[0]} 退出 {r.returncode}")
        return r.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("git 不在 PATH")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git {args[0]} 超时")


def gh(args: list[str]) -> str:
    try:
        r = subprocess.run(["gh", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=60)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or "").strip() or f"gh {args[0]} 退出 {r.returncode}")
        return r.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("gh 不在 PATH")


def stamp() -> str:
    d = datetime.now()
    return f"{d.year}{d.month:02d}{d.day:02d}-{d.hour:02d}{d.minute:02d}"


def read_text(p: str) -> str | None:
    try:
        return Path(p).read_text(encoding="utf-8")
    except Exception:
        return None


def trunc(s, n: int) -> str:
    x = "" if s is None else str(s)
    return x if len(x) <= n else x[:n] + "…[trunc]"


def git_diff_stat() -> str:
    try:
        out = git(["diff", "--stat"])
        lines = [l for l in out.splitlines() if l.strip()]
        return " | ".join(lines[-3:])[:200]
    except Exception:
        return ""


def classify_test_result(text: str) -> str | None:
    """识别 python tests 结果红/绿（文本 fallback）。unittest：OK/FAILED；pytest：passed/failed。"""
    t = (text or "").lower()
    has_fail = bool(re.search(r"failed|failing|error|traceback|\bfail\s+[1-9]", t))  # fail 0 不算红
    has_pass = bool(re.search(r"ok\b|passing|passed|\bpass\s+[1-9]", t))
    if has_fail: return "red"
    if has_pass: return "green"
    return None


def classify_test_exit(tool_result: ToolResultBlock, text: str) -> str | None:
    """判定一次干净 Bash 测试的红/绿。优先 is_error（=退出码：unittest 失败 exit≠0），
    避大输出被截断后文本解析失效。is_error 缺失 → 文本 fallback。"""
    if getattr(tool_result, "is_error", None) is True: return "red"
    if getattr(tool_result, "is_error", None) is False: return "green"
    return classify_test_result(text)


def is_clean_test_cmd(cmd: str) -> bool:
    """判定 Bash command 是否为「干净」的测试运行（SPEC #27：捕获端过滤）。
    跟踪无管道/重定向/链式的裸 python tests/... / python -m unittest|pytest / pytest ——
    这类 tool_result 的 exit code 才可信；带 |>&; 的变体结果无意义。"""
    c = (cmd or "").strip()
    if re.search(r"[|>&;]", c):
        return False
    pat_py = r"^\s*(\S*/)?python\d?(\.\d+)?\s+(tests/|src/.*run_|-\s*m\s+(pytest|unittest)\b)"
    pat_pt = r"^\s*(\S*/)?pytest\b"
    return bool(re.match(pat_py, c) or re.match(pat_pt, c))


def append_run_line(run_log: Path, obj: dict) -> None:
    try:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        sys.stderr.write(f"⚠ run 落盘失败（忽略）: {e}\n")


def create_loop_state(run_log: Path) -> dict:
    return {"run_log": run_log, "turn": 0, "last_test": None, "last_test_cmd": None,
            "no_write_streak": 0, "stalled": False, "pending_test_ids": set()}


async def process_dev_loop(messages, state: dict) -> ResultMessage | None:
    """dev loop 循环体（SPEC #27：监控落盘 + 无进展刹车 + 配对 tool_result 拿红/绿）。对齐 mjs processDevLoop。"""
    result_msg: ResultMessage | None = None
    async for msg in messages:
        if isinstance(msg, AssistantMessage):
            state["turn"] += 1
            blocks = msg.content or []
            has_write = False
            tool_uses = []
            for b in blocks:
                if isinstance(b, TextBlock):
                    sys.stderr.write(f"[dev] {b.text}\n")
                elif isinstance(b, ToolUseBlock):
                    name = b.name or "?"
                    inp = b.input or {}
                    target = (inp.get("file_path") or inp.get("path") or inp.get("notebook_path")
                              or (inp.get("command", "").split("\n")[0] if isinstance(inp.get("command"), str) else "")
                              or "")
                    tool_uses.append({"name": name, "target": trunc(target, 120),
                                      "input": trunc(json.dumps(inp, ensure_ascii=False), INPUT_TRUNC)})
                    if name in WRITE_TOOLS: has_write = True
                    if name == "Bash" and isinstance(inp.get("command"), str):
                        if is_clean_test_cmd(inp["command"].strip()):
                            state["pending_test_ids"].add(b.id)   # 等下个 user msg 配对 tool_result
                            state["last_test_cmd"] = inp["command"].strip()   # 记最后一次干净 test 命令，上报给 dispatch 重放
            if has_write: state["no_write_streak"] = 0
            elif state["last_test"] == "red": state["no_write_streak"] += 1
            append_run_line(state["run_log"], {
                "turn": state["turn"], "tool_use": tool_uses, "diff_stat": git_diff_stat(),
                "test": state["last_test"], "verified_red": state["last_test"] == "red",
                "no_write_streak": state["no_write_streak"],
            })
            if state["last_test"] == "red" and state["no_write_streak"] >= N_STALL:
                state["stalled"] = True
                sys.stderr.write(f"🧯 stalled：验证红后连续 {state['no_write_streak']} 轮无写类进展，主动刹车\n")
                break
        elif isinstance(msg, UserMessage):
            blocks = msg.content or []
            if blocks:
                sys.stderr.write(f"[dev] ← user msg（{len(blocks)} blocks）\n")
            for b in blocks:
                if isinstance(b, ToolResultBlock) and b.tool_use_id in state["pending_test_ids"]:
                    txt = b.content if isinstance(b.content, str) else json.dumps(b.content or "", ensure_ascii=False)
                    res = classify_test_exit(b, txt)
                    if res:
                        state["last_test"] = res
                        sys.stderr.write(f"[dev] test → {res}（is_error={getattr(b,'is_error',None)}, {len(txt)} chars）\n")
                    else:
                        sys.stderr.write(f"[dev] test 结果未识别（is_error={getattr(b,'is_error',None)}, {len(txt)} chars）\n")
                    state["pending_test_ids"].discard(b.tool_use_id)
        elif isinstance(msg, ResultMessage):
            result_msg = msg
    return result_msg


def build_env_for_sdk() -> dict:
    """构造 SDK 子进程 env：把 conda env 的 python 目录 + ~/.local/bin（claude CLI）前置进 PATH，
    使 agent 的裸 `python tests/...` 落在 ashare env（baostock/akshare/sklearn 可用），且 SDK 找得到 claude CLI。"""
    env = dict(os.environ)
    prefix = []
    py_dir = str(Path(sys.executable).resolve().parent)   # dev-agent.py 由 env python 启动 → 其 bin 目录
    if py_dir: prefix.append(py_dir)
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin: prefix.append(local_bin)
    env["PATH"] = os.pathsep.join(prefix) + os.pathsep + env.get("PATH", "")
    return env


def build_prompt(args: dict, prd_text: str, branch: str | None) -> str:
    base = args["base"]
    dry = args["dry_run"]
    head = (f"你是本仓（ashare-llm-analyst）的自治 dev agent。**dry-run 模式**：在当前工作区跑，"
            f"不切分支 / 不 commit / 不 push，改动留工作树供 review。"
            if dry else
            f"你是本仓（ashare-llm-analyst）的自治 dev agent。当前分支 {branch}（基点 {base}）。")
    prot = "" if dry else f"主干 {base} 有 branch protection——你永远只在这个 feature 分支上干活，绝不直推主干。"
    source_block = ""
    if args["source"]:
        src = read_text(args["source"]) or "(读不到 source)"
        source_block = f"\n\n## 触发信号（来自控制面 pa-radar）\n```\n{src}\n```"
    return "\n".join([
        head,
        prot,
        "",
        "## 你的任务（PRD）",
        prd_text,
        source_block,
        "",
        "## 自治守则（详见仓库根 CLAUDE.md）",
        "1. 只在当前分支、本仓范围内改；新工作落 `src/`（版本化分层核心）。`out/`、`back/`、`deep-search/` 是更早的独立脚本，仅为参考、不要在里面改。",
        "2. 改完必须跑仓库的测试——**怎么测你自治决定**（读 CLAUDE.md / 探 tests/；本仓当前入口 `python tests/run_tests.py`，若发现 harness 本身坏了——如 import 路径 / 缺依赖——先修 harness 再跑）。**绿才算完事**：红（test 失败 / import 错 / 取数 raise）= 没做完，修到绿或回滚，绝不留红提交。",
        "3. 遵循既有风格（见 CLAUDE.md）；依赖只通过 requirement.txt / environment.yml 管理（环境名 ashare-llm-analyst）。",
        "4. commit 用 conventional commits，只 commit 与 PRD 相关的改动。",
        "5. 不要 push、不要开 PR——push/PR 由本脚本在你停下后代办。",
        "6. 不碰 .github/ branch protection、不 force push、不删分支、不 pip publish、不动 training_data/ market_data/ downloads/（数据产物）。",
        "",
        "## 何时用子代理分工（Agent 工具）",
        "满足任一才考虑分工，否则单干（子代理有独立 context 成本，别为分工而分工）：",
        "- PRD 横跨多个独立关注点（新增功能 + 补测试 + 审类型，可拆开）；",
        "- 估摸单干要 30+ turn 或读改 5+ 文件。",
        "",
        "分工纪律：用 Agent 工具 spawn 子代理，每个单一职责；给子代理明确目标 + 限定文件范围；",
        "子代理产出回你整合 + 跑仓库测试验证整体；commit/push 只由你（parent）守。",
        "",
        "现在：读 PRD → 规划 → 改代码（落 src/）→ 跑仓库测试 → 到「可提交且 test 绿」即停。",
        "停下前用一段话总结：改了什么 / test 结果 / 遗留风险。",
    ])


async def main() -> int:
    args = parse_args(sys.argv[1:])
    if args["help"]:
        print(HELP); return 0
    if not args["prd"]:
        sys.stderr.write("✗ 缺 --prd（控制面投递的 PRD）\n" + HELP + "\n"); return 10

    prd_text = read_text(args["prd"])
    if prd_text is None:
        sys.stderr.write(f"✗ 读不到 PRD: {args['prd']}\n"); return 10

    base = args["base"]
    slug = re.sub(r"[^a-z0-9]+", "-", Path(args["prd"]).stem.lower()).strip("-")[:24]
    branch = f"{args['branch_prefix']}/{stamp()}-{slug}"

    # 1. 建 feature 分支（dry-run 不切，改动留工作树）
    if not args["dry_run"]:
        try:
            git(["checkout", "-b", branch, base])
        except Exception as e:
            sys.stderr.write(f"✗ 建分支失败: {e}\n"); return 13

    # 2. 组 prompt
    prompt = build_prompt(args, prd_text, None if args["dry_run"] else branch)

    # 3. SDK dev loop（acceptEdits + settingSources + allowedTools，对齐 mjs / SPEC §决策#23）
    run_log = STATE_RUNS_DIR / f"{branch.replace('/', '-')}-{stamp()}.jsonl"
    state = create_loop_state(run_log)
    result_msg: ResultMessage | None = None
    try:
        options = ClaudeAgentOptions(
            cwd=str(REPO_ROOT),
            # model 刻意省略 → 走 roc 代理默认（glm-5.2）；勿传裸 Anthropic model id
            permission_mode="acceptEdits",          # 编辑类自动过（fail-safe，摩擦≈bypass 但不裸放）
            setting_sources=["project"],            # 加载仓 CLAUDE.md(dev 守则) + .claude/hooks
            allowed_tools=["Read", "Grep", "Glob", "Edit", "Write", "MultiEdit",
                           "TodoWrite", "Bash", "Agent"],   # Agent = 子代理分工放行（SPEC §决策#29）
            max_turns=150,
            max_budget_usd=MAX_BUDGET,               # 预算刹车（降级兜底）
            env=build_env_for_sdk(),                 # PATH 前置 conda env python + claude CLI
        )
        result_msg = await process_dev_loop(query(prompt=prompt, options=options), state)
    except Exception as e:
        sys.stderr.write(f"✗ SDK dev loop 异常: {e}\n"); return 11

    cost = getattr(result_msg, "total_cost_usd", None) if result_msg else None
    turns = getattr(result_msg, "num_turns", None) if result_msg else state["turn"]

    if cost in (None, 0):
        sys.stderr.write(f"⚠ 预算刹车未生效（total_cost_usd={cost}），仅 maxTurns 兜底\n")
    else:
        sys.stderr.write(f"💰 本次 cost=${cost}（maxBudgetUsd={MAX_BUDGET}）\n")

    # stalled：不 commit/不开 PR，吐 JSON + exit 12（SPEC #27；半成品靠 run_log 留痕）
    if state["stalled"]:
        print(json.dumps({"ok": False, "stalled": True, "branch": branch, "base": base,
                          "run_log": str(run_log), "cost": cost, "turns": turns, "test_cmd": state.get("last_test_cmd"), "test_passed": state.get("last_test") == "green"}, ensure_ascii=False))
        return 12

    if result_msg and result_msg.is_error:
        sys.stderr.write(f"✗ dev loop 返回错误: {result_msg.result}\n"); return 11

    # 4. dry-run：到此为止
    if args["dry_run"]:
        print(json.dumps({"ok": True, "dry_run": True, "branch": branch, "base": base,
                          "cost": cost, "turns": turns, "test_cmd": state.get("last_test_cmd"), "test_passed": state.get("last_test") == "green", "run_log": str(run_log),
                          "result": getattr(result_msg, "result", None) if result_msg else None},
                         ensure_ascii=False))
        return 0

    # 5. commit + push + 开 PR（branch protection 要求走 PR）
    try:
        git(["add", "-A"])
        diff_stat = git(["diff", "--cached", "--stat"])
        if not diff_stat:
            print(json.dumps({"ok": True, "no_changes": True, "branch": branch,
                              "base": base, "cost": cost, "turns": turns, "test_cmd": state.get("last_test_cmd"), "test_passed": state.get("last_test") == "green"}, ensure_ascii=False))
            return 0
        title = Path(args["prd"]).stem
        git(["commit", "-m", f"feat(pa-dev): {title}",
             "-m", f"项目推进流水线 dev-agent.py 自治产出（基点 {base}）。"])
        git(["push", "-u", "origin", branch])
        pr_url = gh(["pr", "create", "--base", base, "--head", branch,
                     "--title", f"pa-dev: {title}",
                     "--body", "自治 dev agent 产出（PRD 见任务投递）。验证闸（dev-agent 自治）已绿。"])
        print(json.dumps({"ok": True, "branch": branch, "base": base, "pr_url": pr_url,
                          "cost": cost, "turns": turns, "test_cmd": state.get("last_test_cmd"), "test_passed": state.get("last_test") == "green"}, ensure_ascii=False))
        return 0
    except Exception as e:
        sys.stderr.write(f"✗ git/push/PR 阶段失败: {e}\n")
        print(json.dumps({"ok": False, "error": str(e), "branch": branch,
                          "base": base, "cost": cost, "turns": turns, "test_cmd": state.get("last_test_cmd"), "test_passed": state.get("last_test") == "green"}, ensure_ascii=False))
        return 13


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(99)
    except Exception as e:
        sys.stderr.write(f"✗ 未捕获异常: {e}\n")
        sys.exit(99)
