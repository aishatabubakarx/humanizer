import os
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

    def generate(self, prompt: str, temperature: float = 0.9, system_instruction: str = None) -> str:
        """
        Generates text using Google Gemini.
        """
        if not self.client:
            env_key = os.environ.get("GEMINI_API_KEY")
            if env_key:
                self.client = genai.Client(api_key=env_key)
            else:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")

        if not system_instruction:
            system_instruction = (
                "You are an expert human editor. Rewrite the text to sound completely "
                "natural, varying sentence lengths and eliminating robotic AI phrasing."
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )

        return response.text.strip()


def call_llm(prompt: str, temperature: float = 0.9, system_instruction: str = None) -> str:
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
