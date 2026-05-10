import json
import os
from openai import OpenAI


def _get_config() -> tuple[OpenAI, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("设置环境变量: ANTHROPIC_API_KEY 或 OPENAI_API_KEY")

    # Auto-detect provider by key prefix
    if api_key.startswith("sk-ant"):
        # Anthropic native key — use OpenAI-compatible Anthropic endpoint
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    else:
        # OpenAI-format key (DeepSeek, OpenAI, etc.)
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        model = os.environ.get("OPENAI_MODEL", "deepseek-v4-pro")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def ask(prompt: str, system: str = "", model: str | None = None) -> str:
    client, default_model = _get_config()
    model = model or default_model
    # Auto-redirect: if using Claude model name but on non-Anthropic backend
    if model.startswith("claude-") and "anthropic" not in str(client.base_url):
        model = default_model
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


def extract_json(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences and extra text."""
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    response = response.strip()
    if response.startswith("["):
        depth = 0
        end = 0
        for i, ch in enumerate(response):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        response = response[:end]
    elif response.startswith("{"):
        depth = 0
        end = 0
        for i, ch in enumerate(response):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        response = response[:end]
    return response


def ask_json(prompt: str, system: str = "", model: str | None = None) -> dict | list:
    """Ask LLM and parse JSON response with robust error handling."""
    response = ask(prompt, system=system, model=model)
    json_str = extract_json(response)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        retry_prompt = prompt + "\n\nIMPORTANT: Your previous response was not valid JSON. Output ONLY valid JSON this time, no other text."
        response = ask(retry_prompt, system=system, model=model)
        json_str = extract_json(response)
        return json.loads(json_str)
