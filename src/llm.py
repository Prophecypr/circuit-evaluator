import json
import os
from openai import OpenAI


def _get_config() -> tuple[OpenAI, str]:
    # Priority: Ollama > Anthropic > DeepSeek/OpenAI
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "")
    if ollama_url:
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
        client = OpenAI(api_key="ollama", base_url=ollama_url)
        return client, model

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        raise RuntimeError("设置 OLLAMA_BASE_URL 或 API_KEY 环境变量")

    if api_key.startswith("sk-ant"):
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        model = os.environ.get("OPENAI_MODEL", "deepseek-v4-pro")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def ask(prompt: str, system: str = "", model: str | None = None) -> str:
    client, default_model = _get_config()
    model = model or default_model
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.0,
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
