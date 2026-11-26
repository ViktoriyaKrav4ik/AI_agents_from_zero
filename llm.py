"""
Заняття 3. Функція llm(): один контракт, кілька провайдерів.

Лише стандартна бібліотека Python: запити йдуть напряму в REST API
провайдерів через urllib, тому нічого встановлювати не треба.

Запуск:
    python llm.py keys                      # крок 1: перевірити ключі (друкує лише останні 4 символи)
    python llm.py hello                     # крок 2: перший виклик до кожного провайдера
    python llm.py hello --provider google   #         або лише до одного
    python llm.py retry-demo                # крок 4: навмисно невірний ключ -> видно повтори і фінальну помилку
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- налаштування

PROVIDERS = {
    # провайдер: модель за замовчуванням, змінна з ключем, змінна, якою модель
    # можна замінити в .env (наприклад GOOGLE_MODEL=gemini-3.5-flash-lite)
    "google": {
        # gemini-2.5-flash новим користувачам уже недоступна (HTTP 404)
        "model": "gemini-3.8-flash",
        "key_env": "GOOGLE_API_KEY",
        "model_env": "GOOGLE_MODEL",
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
    },
    "anthropic": {
        "model": "claude-haiku-5-5",
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": "ANTHROPIC_MODEL",
    },
}

# Ціни $ за 1 млн токенів (вхід, вихід) з офіційних прайсів — див. results.md, «Прайси».
# Нова модель без ціни тут не запуститься: вартість не рахується «на око».
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),       # платний тариф до 31.12.2026, вихід включно з «мисленням»
    "gemini-3.5-flash-lite": (0.30, 2.50),  # платний тариф, вихід включно з «мисленням»
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-haiku-5-5": (0.10, 0.50),
}


def model_for(provider: str) -> str:
    cfg = PROVIDERS[provider]
    return os.environ.get(cfg["model_env"]) or cfg["model"]

# Відповідь — короткий JSON (~15 токенів), але в Gemini сюди ж входить «мислення»,
# тому ліміт із запасом. Якщо stop_reason покаже обрізання
# (MAX_TOKENS / length / max_tokens) — збільшити.
MAX_TOKENS = 2048

MAX_ATTEMPTS = 5          # 1 спроба + 4 повтори
BACKOFF_START = 1         # паузи між спробами: 1 -> 2 -> 4 -> 8 с
HTTP_TIMEOUT = 60         # секунд на один HTTP-запит


# ---------------------------------------------------------------- .env

def load_env(path: str = ".env") -> None:
    """Читає .env поруч зі скриптом і кладе значення в os.environ.

    Значення з .env перекривають уже встановлені змінні, щоб після зміни
    файлу не лишився старий ключ із середовища.
    """
    full = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(full):
        return
    with open(full, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ[name.strip()] = value.strip().strip('"').strip("'")


def get_key(provider: str) -> str:
    env = PROVIDERS[provider]["key_env"]
    key = os.environ.get(env, "")
    if not key:
        raise RuntimeError(f"Немає ключа {env} у .env")
    return key


def mask(key: str) -> str:
    return "…" + key[-4:] if key else "(порожньо)"


# ---------------------------------------------------------------- HTTP

def post_json(url: str, headers: dict, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:  # дістаємо лише текст помилки з JSON-відповіді
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        err = RuntimeError(f"HTTP {e.code}: {' '.join(str(detail).split())[:300]}")
        err.status = e.code
        raise err from None


# ---------------------------------------------------------------- гілки провайдерів
# Кожна гілка повертає (text, in_tokens, out_tokens, stop_reason).

def _call_google(prompt, system, model, key, max_tokens):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            # gemini-3.8-flash не дає вимкнути «мислення» зовсім; low — найменший рівень.
            # Токени мислення входять у maxOutputTokens і оплачуються як вихідні.
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    if system:
        # у Gemini system — окреме поле, не роль у contents
        body["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = post_json(url, {"x-goog-api-key": key}, body)
    cand = r["candidates"][0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    u = r.get("usageMetadata", {})
    # токени «роздумів», якщо є, оплачуються як вихідні
    out_tok = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return text, u.get("promptTokenCount", 0), out_tok, cand.get("finishReason")


def _call_openai(prompt, system, model, key, max_tokens):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_completion_tokens": max_tokens}
    r = post_json("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {key}"}, body)
    choice = r["choices"][0]
    u = r.get("usage", {})
    return (choice["message"].get("content") or "", u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0), choice.get("finish_reason"))


def _call_anthropic(prompt, system, model, key, max_tokens):
    body = {
        "model": model,
        "max_tokens": max_tokens,               # обов'язковий, без нього 400
        "messages": [{"role": "user", "content": prompt}],
        # API-версія 2026-10-21 не приймає temperature (400).
        # Еквівалент для відтворюваності — вимкнене мислення.
        "thinking": {"type": "disabled"},
    }
    if system:
        body["system"] = system                 # system — окреме поле, не в messages
    r = post_json("https://api.anthropic.com/v1/messages",
                  {"x-api-key": key, "anthropic-version": "2026-10-21"}, body)
    text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    u = r.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0), r.get("stop_reason")


_BRANCHES = {"google": _call_google, "openai": _call_openai, "anthropic": _call_anthropic}


# ---------------------------------------------------------------- головна функція

def llm(prompt: str, system: str = "", provider: str = "google",
        max_tokens: int = MAX_TOKENS) -> dict:
    """
    Повертає словник:
    {
      "text": "...",          # відповідь моделі
      "in_tokens": 412,
      "out_tokens": 87,
      "stop_reason": "end_turn",
      "seconds": 1.8,         # час успішної спроби
      "cost_usd": 0.00042
    }
    Після MAX_ATTEMPTS невдалих спроб піднімає RuntimeError (не повертає None).
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Невідомий провайдер {provider!r}. Є: {', '.join(PROVIDERS)}")
    model = model_for(provider)
    if model not in PRICES:
        raise ValueError(f"Немає ціни для моделі {model!r}: додайте її в PRICES з офіційного прайсу")
    price_in, price_out = PRICES[model]
    call = _BRANCHES[provider]

    delay = BACKOFF_START
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            key = get_key(provider)   # читаємо щоразу: ключ з .env, у коді його немає
            t0 = time.perf_counter()
            text, in_tok, out_tok, stop = call(prompt, system, model, key, max_tokens)
            seconds = time.perf_counter() - t0
            break
        except Exception as e:  # noqa: BLE001
            # Для навчальної демонстрації повторюємо будь-яку помилку, зокрема 401
            # від невірного ключа. У робочому коді 400/401/403 повторювати не варто,
            # лише 429, 5xx і мережеві збої.
            last_error = e
            if getattr(e, "status", None) == 404:
                # модель або адреса не існує — повтор не допоможе
                raise RuntimeError(f"[{provider}] {e}") from e
            if attempt == MAX_ATTEMPTS:
                print(f"[{provider}] спроба {attempt}/{MAX_ATTEMPTS} невдала: {e}", file=sys.stderr)
                raise RuntimeError(
                    f"[{provider}] усі {MAX_ATTEMPTS} спроб невдалі. Остання помилка: {last_error}"
                ) from last_error
            print(f"[{provider}] спроба {attempt}/{MAX_ATTEMPTS} невдала: {e}. "
                  f"Повтор через {delay} с", file=sys.stderr)
            time.sleep(delay)
            delay *= 2

    # вартість — одна формула для всіх провайдерів
    cost = (in_tok * price_in + out_tok * price_out) / 1_000_000
    return {
        "text": text,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "stop_reason": stop,
        "seconds": round(seconds, 2),
        "cost_usd": round(cost, 8),
        "provider": provider,
        "model": model,
    }


# ---------------------------------------------------------------- CLI

def _providers_from_args(args):
    if "--provider" in args:
        return [args[args.index("--provider") + 1]]
    return [p for p in PROVIDERS if os.environ.get(PROVIDERS[p]["key_env"])]


def main(argv):
    load_env()
    cmd = argv[1] if len(argv) > 1 else "hello"

    if cmd == "keys":
        for p, cfg in PROVIDERS.items():
            print(f"{cfg['key_env']:<18} {mask(os.environ.get(cfg['key_env'], ''))}")

    elif cmd == "hello":
        for p in _providers_from_args(argv):
            r = llm("Привіт! Відповідай одним реченням: що таке API?", provider=p)
            print(f"\n=== {p} / {r['model']} ===")
            print("text:       ", r["text"].strip())
            print("stop_reason:", r["stop_reason"])
            print("tokens:      in", r["in_tokens"], "/ out", r["out_tokens"])
            print(f"seconds:     {r['seconds']}   cost_usd: {r['cost_usd']:.8f}")

    elif cmd == "retry-demo":
        p = _providers_from_args(argv)[0] if "--provider" in argv else "google"
        os.environ[PROVIDERS[p]["key_env"]] = "invalid-key-for-retry-demo"
        print(f"Демо retry: {p} з навмисно невірним ключем {mask(os.environ[PROVIDERS[p]['key_env']])}")
        llm("Привіт", provider=p)   # має впасти з RuntimeError після 5 спроб

    else:
        print(__doc__)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except RuntimeError as e:
        print(f"\nПОМИЛКА: {e}", file=sys.stderr)
        sys.exit(1)
