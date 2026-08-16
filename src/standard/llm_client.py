import os
from google import genai
from google.genai import types

class LLMClient:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        Initializes the client using Google's official GenAI SDK.
        It uses the provided API key or checks system environment variables.
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
            # Fallback check if environment variable was set after init
            env_key = os.environ.get("GEMINI_API_KEY")
            if env_key:
                self.client = genai.Client(api_key=env_key)
            else:
                raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment or secrets.")

        if not system_instruction:
            system_instruction = (
                "You are an expert editor. Rewrite the text to sound completely "
                "natural and human, varying sentence structures and removing AI tropes."
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
    Standalone helper function for quick API calls.
    """
    client = LLMClient()
    return client.generate(prompt=prompt, temperature=temperature, system_instruction=system_instruction)
