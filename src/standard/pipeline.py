"""Standard Pipeline (v1.5.1) — production path.

Executes the 4-step humanization chain:
Step 1: LLM Restructure (Pass 1 - Gemini)
Step 2: LLM Polish (Pass 2 - Gemini)
Step 3: Japanese -> Finnish (Google Translate)
Step 4: Finnish -> English (Azure Translator)
"""

import logging
import click
import toml
from typing import Dict, Any, Optional

from .llm_client import resolve_llm_config
from .llm_rewriter import llm_rewrite
from .translators import google_translate, azure_translate

logger = logging.getLogger(__name__)


def run_standard_pipeline(
    text: str,
    config: Dict[str, Any],
    target_lang: str = "en",
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the 4-step humanization pipeline on input text."""
    if not text or not text.strip():
        return {
            "result": "",
            "steps": {},
            "status": "empty_input",
        }

    # Resolve LLM client if not explicitly passed
    if client is None or model is None:
        client, model = resolve_llm_config(config)

    steps = {}

    # Step 1 & Step 2: Gemini LLM Rewriting Passes
    try:
        step1_out, step2_out = llm_rewrite(
            text,
            client=client,
            model=model,
        )
        steps["step1_llm_rewrite_1"] = step1_out
        steps["step2_llm_rewrite_2"] = step2_out
    except Exception as e:
        logger.error(f"Error in LLM rewriting steps: {e}")
        raise e

    # Step 3: Intermediate Translation (Japanese -> Finnish)
    try:
        step3_out = google_translate(
            text=step2_out,
            source="ja",
            target="fi",
        )
        steps["step3_ja_to_fi"] = step3_out
    except Exception as e:
        logger.error(f"Error in Step 3 translation (JA -> FI): {e}")
        raise e

    # Step 4: Final Translation (Finnish -> English via Azure)
    try:
        final_out = azure_translate(
            text=step3_out,
            source="fi",
            target=target_lang,
        )
        steps["step4_fi_to_en"] = final_out
    except Exception as e:
        logger.error(f"Error in Step 4 translation (FI -> EN via Azure): {e}")
        raise e

    return {
        "result": final_out,
        "steps": steps,
        "status": "success",
    }


@click.command()
@click.option("--input", "input_text", required=True, help="Input text to humanize")
@click.option("--config", default="config/config.toml", help="Path to config TOML file")
@click.option("--target", default="en", help="Target language code")
def main(input_text: str, config: str, target: str):
    """CLI entry point for running the standard pipeline."""
    cfg = toml.load(config)
    result = run_standard_pipeline(input_text, cfg, target_lang=target)
    print(result["result"])


if __name__ == "__main__":
    main()
