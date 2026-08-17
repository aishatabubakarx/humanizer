"""Translation engines: Google Translate and Azure Translator."""

import os
import httpx
from deep_translator import GoogleTranslator


def google_translate(text: str, source: str = "auto", target: str = "en") -> str:
    """Translate text using Google Translate.

    Args:
        text: Text to translate.
        source: Source language code.
        target: Target language code.

    Returns:
        Translated text.
    """
    if not text or not text.strip():
        return ""

    translator = GoogleTranslator(source=source, target=target)

    # Handle long texts by chunking (~5000 char limit)
    if len(text) > 4500:
        chunks = _split_text(text, max_len=4500)
        return " ".join(translator.translate(chunk) for chunk in chunks)

    return translator.translate(text)


def azure_translate(text: str, source: str = "fi", target: str = "en", api_key: str = None) -> str:
    """Translate text using Azure Cognitive Services Translator API.

    Args:
        text: Text to translate.
        source: Source language code.
        target: Target language code.
        api_key: Optional Azure Translator API key. If not provided, reads AZURE_TRANSLATOR_KEY from env.

    Returns:
        Translated text.
    """
    if not text or not text.strip():
        return ""

    key = api_key or os.getenv("AZURE_TRANSLATOR_KEY")
    region = os.getenv("AZURE_TRANSLATOR_REGION", "eastus")

    if not key:
        raise ValueError("AZURE_TRANSLATOR_KEY environment variable is not set.")

    url = "https://api.cognitive.microsofttranslator.com/translate"
    params = {
        "api-version": "3.0",
        "from": source,
        "to": target
    }
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-Type": "application/json"
    }
    body = [{"text": text}]

    response = httpx.post(url, params=params, headers=headers, json=body, timeout=60)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and len(data) > 0 and "translations" in data[0]:
        return data[0]["translations"][0]["text"]
    else:
        raise RuntimeError(f"Unexpected Azure Translator response: {data}")


# Backward compatibility alias
niutrans_translate = azure_translate


def _split_text(text: str, max_len: int = 4500) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_len:
            if current:
                chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks if chunks else [text]
