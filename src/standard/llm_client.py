import os
import time
import warnings
# Suppress the GenAI SDK warning globally
warnings.filterwarnings("ignore", category=UserWarning)

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

# Ordered list of models to try. If the first one is overloaded/unavailable
# (503) or hits a rate limit (429), the client automatically falls through
# to the next one in the list.
DEFAULT_MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
]


class LLMClient:
    def __init__(self, api_key: str = None, model: str = None, fallback_models: list = None):
        """
        Initializes the client using Google's official GenAI SDK.
        Reads GEMINI_API_KEY from environment if api_key is not passed explicitly.

        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model: Primary model to try first. Defaults to the first entry
                in fallback_models.
            fallback_models: Ordered list of models to try if the primary
                model fails. Defaults to DEFAULT_MODEL_FALLBACKS.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.fallback_models = fallback_models or list(DEFAULT_MODEL_FALLBACKS)
        self.model = model or self.fallback_models[0]

        # Primary model goes first; remaining fallbacks follow, without duplicates.
        self.models_to_try = [self.model] + [
            m for m in self.fallback_models if m != self.model
        ]

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def _ensure_client(self):
        if not self.client:
            env_key = os.environ.get("GEMINI_API_KEY")
            if env_key:
                self.client = genai.Client(api_key=env_key)
            else:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")

    def generate(
        self,
        prompt: str,
        temperature: float = 1.3,
        system_instruction: str = None,
        history: dict = None,
        retries_per_model: int = 2,
        retry_delay: float = 3.0,
    ) -> str:
        """
        Generates text using Google Gemini, with automatic fallback across
        models if one is overloaded or unavailable.

        For each model in self.models_to_try, retries up to
        `retries_per_model` times on transient server errors (503
        UNAVAILABLE, 429 RESOURCE_EXHAUSTED) before moving on to the next
        model in the list. Raises if every model/attempt fails.
        """
        self._ensure_client()

        if not system_instruction:
            system_instruction = (
                "You are an expert copywriter and multilingual localization "
                "specialist. Rewrite and translate text naturally, removing "
                "any AI-sounding phrasing."
            )

        contents = []
        if history:
            contents.append(
                types.Content(role="user", parts=[types.Part(text=history["input"])])
            )
            contents.append(
                types.Content(role="model", parts=[types.Part(text=history["output"])])
            )
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        last_error = None

        for model_name in self.models_to_try:
            for attempt in range(1, retries_per_model + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    return response.text.strip()
                except ServerError as e:
                    # 5xx: model overloaded / temporarily unavailable. Retry,
                    # then fall through to the next model if retries run out.
                    last_error = e
                    print(f"[LLMClient] {model_name} failed (attempt {attempt}/{retries_per_model}): {e}")
                    if attempt < retries_per_model:
                        time.sleep(retry_delay)
                except ClientError:
                    # 4xx: bad request, auth, etc. Retrying or switching
                    # models won't fix these, so surface immediately.
                    raise

        raise RuntimeError(
            f"All Gemini models failed: {self.models_to_try}. Last error: {last_error}"
        )


def call_llm(prompt: str, temperature: float = 1.3, system_instruction: str = None) -> str:
    """
    Helper function for standalone LLM calls.
    """
    client = LLMClient()
    return client.generate(prompt=prompt, temperature=temperature, system_instruction=system_instruction)


def resolve_llm_config(config=None):
    """
    Bridge function expected by pipeline.py.
    Accepts optional config dictionary argument. Reads an optional
    `[llm].fallback_models` list from config.toml if present.
    """
    fallback_models = None
    if config:
        fallback_models = config.get("llm", {}).get("fallback_models")
    client = LLMClient(fallback_models=fallback_models)
    return client, client.model
