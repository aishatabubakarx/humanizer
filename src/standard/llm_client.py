import os
import warnings
# Suppress the GenAI SDK warning globally
warnings.filterwarnings("ignore", category=UserWarning)

from google import genai
from google.genai import types


class LLMClient:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        Initializes the client using Google's official GenAI SDK.
        Reads GEMINI_API_KEY from environment if api_key is not passed explicitly.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model

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
    ) -> str:
        """
        Generates text using Google Gemini.

        Args:
            prompt: The current user prompt to send.
            temperature: Sampling temperature (1.3 mirrors the original
                DeepSeek pipeline's humanization temperature).
            system_instruction: Optional system prompt.
            history: Optional dict with 'input' and 'output' keys from a
                previous round. Mirrors the original pipeline's 1-round
                history, where Step 2 sees Step 1's turn as context.
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

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )

        return response.text.strip()


def call_llm(prompt: str, temperature: float = 1.3, system_instruction: str = None) -> str:
    """
    Helper function for standalone LLM calls.
    """
    client = LLMClient()
    return client.generate(prompt=prompt, temperature=temperature, system_instruction=system_instruction)


def resolve_llm_config(config=None):
    """
    Bridge function expected by pipeline.py.
    Accepts optional config dictionary argument.
    """
    client = LLMClient()
    return client, client.model
