import json
import os
import re

from anthropic import Anthropic


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    return Anthropic(api_key=api_key)


def ask(prompt: str, system: str = "", model: str = "claude-sonnet-4-6") -> str:
    client = get_client()
    kwargs = {"model": model, "max_tokens": 4096}
    messages = [{"role": "user", "content": prompt}]
    if system:
        kwargs["system"] = system
    resp = client.messages.create(messages=messages, **kwargs)
    return resp.content[0].text


def extract_json(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences and extra text."""
    response = response.strip()
    # Remove markdown code fences
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove opening fence (may have language tag) and closing fence
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    # Try to find JSON array or object bounds
    response = response.strip()
    if response.startswith("["):
        # Find matching closing bracket
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


def ask_json(prompt: str, system: str = "", model: str = "claude-sonnet-4-6") -> dict | list:
    """Ask LLM and parse JSON response with robust error handling."""
    response = ask(prompt, system=system, model=model)
    json_str = extract_json(response)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Retry once with stricter prompt
        retry_prompt = prompt + "\n\nIMPORTANT: Your previous response was not valid JSON. Output ONLY valid JSON this time, no other text."
        response = ask(retry_prompt, system=system, model=model)
        json_str = extract_json(response)
        return json.loads(json_str)
