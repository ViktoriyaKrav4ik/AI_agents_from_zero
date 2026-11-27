"""
Крок 5–6. Прогін тестсету з ДЗ заняття 2 через llm() на двох провайдерах.

Вхід:
    prompts/prompt_v2.txt   — промпт-переможець v2 з ДЗ 2 (6 блоків), місце для входу {{message}}
    testset.json            — 5 входів: [{"id": 1, "input": "...", "expected": {...}}, ...]
                              expected — поля, які мають збігтися у JSON-відповіді

Запуск:
    python run_testset.py                       # google + openai
    python run_testset.py google anthropic      # будь-які два (або більше) провайдери

Вихід: results.json (усі відповіді) і таблиця Markdown у консолі для results.md.
"""

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # щоб llm.py знаходився і при запуску з іншої папки

from llm import PROVIDERS, llm, load_env, model_for  # noqa: E402
PAUSE_BETWEEN_CALLS = 6   # с; безкоштовний тариф Gemini має ліміт запитів на хвилину


def load_prompt(path: Path):
    """prompt_v2.txt з ДЗ 2 без змін: блоки від «Роль» до «Формат виводу» -> system,
    блок «Дані» з місцем {{message}} -> повідомлення користувача."""
    text = path.read_text(encoding="utf-8")
    head, sep, tail = text.rpartition("\nДані\n")
    if not sep:
        return "", text                      # блоку «Дані» немає: увесь промпт як user
    return head.strip(), "Дані\n" + tail.strip()


def parse_json(text: str):
    """Відповідь має парситись у Python. Знімаємо ```json ... ``` огортку, якщо є."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return json.loads(t)


def norm(v):
    # усе порівнюємо як рядки: "0048" і 48 — різні відповіді; null лишається None
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none")):
        return None
    return str(v).strip().lstrip("№").strip().lower()


def is_correct(answer: dict, expected: dict) -> bool:
    """Правильно, якщо кожне поле з expected збігається з полем відповіді."""
    return all(norm(answer.get(k)) == norm(v) for k, v in expected.items())


def main(providers):
    load_env()
    system, user_tpl = load_prompt(HERE / "prompts" / "prompt_v2.txt")
    tests = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))

    rows, summary = [], {}
    for p in providers:
        s = summary[p] = {"model": model_for(p), "ok": 0, "parsed": 0,
                          "in": 0, "out": 0, "cost": 0.0, "secs": []}
        for t in tests:
            prompt = user_tpl.replace("{{message}}", t["input"])
            r = llm(prompt, system=system, provider=p)
            try:
                answer = parse_json(r["text"])
                parsed = True
            except (json.JSONDecodeError, ValueError):
                answer, parsed = None, False
            ok = parsed and isinstance(answer, dict) and is_correct(answer, t.get("expected", {}))

            s["ok"] += ok
            s["parsed"] += parsed
            s["in"] += r["in_tokens"]
            s["out"] += r["out_tokens"]
            s["cost"] += r["cost_usd"]
            s["secs"].append(r["seconds"])
            rows.append({"test_id": t["id"], **r, "parsed": parsed, "answer": answer, "correct": ok})

            print(f"[{p}] test {t['id']}: in={r['in_tokens']} out={r['out_tokens']} "
                  f"stop={r['stop_reason']} {r['seconds']}s ${r['cost_usd']:.6f} "
                  f"json={'так' if parsed else 'НІ'} правильно={'так' if ok else 'ні'}")
            time.sleep(PAUSE_BETWEEN_CALLS)

    (HERE / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                       encoding="utf-8", newline="")

    n = len(tests)
    print("\nпровайдер | модель | правильних із", n, "| in tok | out tok | вартість $ | сер. час, с")
    print("----------|--------|-----------------|--------|---------|------------|------------")
    for p, s in summary.items():
        avg = sum(s["secs"]) / len(s["secs"])
        print(f"{p} | {s['model']} | {s['ok']} | {s['in']} | {s['out']} | {s['cost']:.6f} | {avg:.2f}")
    print(f"\nУсього викликів: {len(rows)}. Деталі — results.json")


if __name__ == "__main__":
    main(sys.argv[1:] or ["google", "openai"])
