from __future__ import annotations

import os
from typing import Any


# Central model routing used by generation, online judges, and final evaluation.
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "Qwen/Qwen3-14B": {
        "family": "qwen",
    },
    "deepseek-v4-flash": {
        "family": "deepseek",
    },
    "deepseek-v4-pro": {
        "family": "deepseek",
    },
    "gpt-4o-mini": {
        "family": "gpt",
    },
}


def _infer_family(model_name: str | None) -> str:
    lower_name = str(model_name or "").strip().lower()
    if "deepseek" in lower_name:
        return "deepseek"
    if "qwen" in lower_name:
        return "qwen"
    if lower_name.startswith("gpt"):
        return "gpt"
    return "generic"


MODEL_ALIASES = {
    "qwen": "Qwen/Qwen3-14B",
    "deepseek": "deepseek-v4-flash",
    "deepseekpro": "deepseek-v4-pro",
    "gpt4omini": "gpt-4o-mini",
}


def _canonical_model_name(model_name: str | None) -> str:
    name = str(model_name or "").strip()
    if not name:
        return ""
    alias = MODEL_ALIASES.get(name.lower())
    if alias:
        return alias
    for registered_name in MODEL_PRESETS:
        if registered_name.lower() == name.lower():
            return registered_name
    return name


def resolve_model(
    model_name: str | None,
    *,
    key_env: str | None = None,
    role: str = "generation",
) -> dict[str, str | None]:
    """Resolve a model using its preset and the generation/judge environment.

    The URL may be empty during offline setup; API entry points require it.
    """

    name = _canonical_model_name(model_name)
    if role not in {"generation", "judge"}:
        raise ValueError(f"Unsupported model role: {role}")
    prefix = "JUDGE" if role == "judge" else "GEN"
    url = os.environ.get(f"{prefix}_BASE_URL", "").strip()
    preset = MODEL_PRESETS.get(name)
    if preset is None and not url:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(
            f"Unsupported model '{name}'. Set {prefix}_BASE_URL or add it to "
            f"Evoguard/llm.py:MODEL_PRESETS. Available models: {supported}"
        )
    preset = preset or {}
    route = {
        "model": name,
        "family": str(preset.get("family") or _infer_family(name)),
        "url": url,
        "key": os.environ.get(f"{prefix}_API_KEY") or None,
    }
    if key_env:
        route["key"] = _read_env_key(key_env, "model key override")
    if url and not url.startswith("https://"):
        raise ValueError(f"{prefix}_BASE_URL must use HTTPS.")
    return route


def model_family(model_name: str | None) -> str:
    name = _canonical_model_name(model_name)
    preset = MODEL_PRESETS.get(name)
    if preset:
        return str(preset.get("family", "generic"))
    return _infer_family(name)


def preset_for(model_name: str | None, *, role: str = "generation") -> dict[str, str | None]:
    route = resolve_model(model_name, role=role)
    return {
        "family": route["family"],
        "url": route["url"],
        "key": route["key"],
    }


def _arg_value(args, name: str) -> str:
    return str(getattr(args, name, "") or "").strip()


def _read_env_key(env_name: str, option_name: str) -> str:
    env_name = str(env_name or "").strip()
    if not env_name:
        return ""
    value = str(os.environ.get(env_name, "") or "").strip()
    if not value:
        raise ValueError(f"{option_name} was set to '{env_name}', but that environment variable is empty or missing.")
    return value


def apply_key_env_overrides(config, args):
    """Apply temporary key overrides after model presets have selected URLs/models."""

    llm = config.llm
    generation_key = _read_env_key(getattr(args, "generation_key_env", None), "--generation-key-env")
    prompt_key = _read_env_key(getattr(args, "prompt_key_env", None), "--prompt-key-env")
    judge_key = _read_env_key(getattr(args, "judge_key_env", None), "--judge-key-env")

    if generation_key:
        llm.generation_key = generation_key
        if not prompt_key and str(llm.prompt_model) == str(llm.generation_model):
            llm.prompt_key = generation_key
    if prompt_key:
        llm.prompt_key = prompt_key
    if judge_key:
        llm.judge_key = judge_key
    return config


def apply_model_overrides(config, args):
    llm = config.llm
    requested_generation = _arg_value(args, "generation_model")
    requested_prompt = _arg_value(args, "prompt_model")
    requested_judge = _arg_value(args, "judge_model")

    generation_model = requested_generation or str(llm.generation_model)
    if requested_prompt:
        prompt_model = requested_prompt
    elif requested_generation:
        prompt_model = generation_model
    else:
        prompt_model = str(llm.prompt_model)
    judge_model = requested_judge or str(llm.judge_model)

    generation_route = resolve_model(generation_model, role="generation")
    prompt_route = resolve_model(prompt_model, role="generation")
    judge_route = resolve_model(judge_model, role="judge")

    llm.generation_model = str(generation_route["model"])
    llm.generation_url = str(generation_route["url"])
    llm.generation_key = generation_route["key"]
    llm.prompt_model = str(prompt_route["model"])
    llm.prompt_url = str(prompt_route["url"])
    llm.prompt_key = prompt_route["key"]
    llm.judge_model = str(judge_route["model"])
    llm.judge_url = str(judge_route["url"])
    llm.judge_key = judge_route["key"]
    apply_key_env_overrides(config, args)
    return config


def build_request_kwargs(
    *,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    request_timeout: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": str(model_name),
        "messages": list(messages),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if request_timeout is not None:
        # OpenAI Python v1 chat.completions.create expects `timeout`, not `request_timeout`.
        kwargs["timeout"] = int(request_timeout)

    if response_format is not None:
        kwargs["response_format"] = response_format

    return kwargs
