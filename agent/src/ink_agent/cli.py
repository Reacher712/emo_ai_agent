import argparse
import json
import sys
from uuid import uuid4

from .service import AgentService


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="国画 Agent LangGraph 骨架（仅 mock）")
    parser.add_argument("--db-dir", default="runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="明确提交创作任务，不进行自动意图识别")
    create.add_argument("message")
    create.add_argument("--scenario", choices=["direct", "ask", "revise", "fallback", "retrieve"], default="direct")
    create.add_argument("--session", default="demo")
    create.add_argument("--parent")
    chat = sub.add_parser("chat")
    chat.add_argument("message")
    chat.add_argument("--session", default="demo")
    for name in ("status", "cancel", "recover"):
        sub.add_parser(name).add_argument("task_id")
    answer = sub.add_parser("answer")
    answer.add_argument("task_id")
    choices = answer.add_mutually_exclusive_group(required=True)
    choices.add_argument("--choice")
    choices.add_argument("--text")
    choices.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    try:
        with AgentService(args.db_dir) as app:
            if args.command == "create":
                task = app.submit(args.message, session_id=args.session, scenario=args.scenario,
                                  parent_artifact_id=args.parent)
                result = app.wait(task)
            elif args.command == "chat":
                result = app.chat(args.message, session_id=args.session)
            elif args.command == "answer":
                current = app.status(args.task_id)
                if current["question"] is None:
                    raise ValueError("Task is not waiting for an answer")
                payload = dict(task_id=args.task_id, question_id=current["question"]["question_id"],
                               message_id=uuid4().hex,
                               answer_type="choice" if args.choice else "free_text" if args.text else "skip",
                               option_id=args.choice, text=args.text)
                result = app.answer(args.task_id, payload)
                if result["status"] == "resuming":
                    result = app.wait(args.task_id)
            else:
                result = getattr(app, args.command)(args.task_id)
                if result["status"] == "recovering":
                    result = app.wait(args.task_id)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
