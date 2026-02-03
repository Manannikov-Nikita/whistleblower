import os
from pathlib import Path
from typing import Optional


class SummarizationError(RuntimeError):
    pass


def _resolve_model(model: Optional[str]) -> str:
    return model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _build_prompt(text: str, language: str) -> str:
    return (
        "Summarize the meeting transcript in Markdown.\n"
        "Output language: " + language + ".\n\n"
        "Requirements:\n"
        "- Provide bullet points per speaker (use the speaker labels as-is).\n"
        "- Provide final decisions/agreements.\n"
        "- Provide items discussed but not decided or deferred.\n"
        "- Keep chronological order within each section where possible.\n"
        "- Do not add timestamps.\n\n"
        "Transcript:\n"
        + text
    )


def summarize_text(
    input_text: str,
    output_path: Optional[Path] = None,
    model: Optional[str] = None,
    language: str = "Russian",
) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SummarizationError(
            "openai is not installed. Run: uv sync"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummarizationError(
            "OPENAI_API_KEY is not set (check .env)."
        )

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    prompt = _build_prompt(input_text, language)
    response = client.responses.create(
        model=_resolve_model(model),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a careful assistant that writes concise, "
                    "structured meeting summaries."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = response.output_text or ""
    if not text.strip():
        raise SummarizationError("OpenAI returned empty response.")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text.strip() + "\n", encoding="utf-8")

    return text
