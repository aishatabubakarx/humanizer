from src.standard.llm_client import LLMClient

class LLMRewriter:
    def __init__(self, api_key: str = None):
        self.client = LLMClient(api_key=api_key)

    def rewrite(self, text: str, temperature: float = 0.9) -> str:
        """
        First pass: Restructure and break rigid AI writing patterns.
        """
        prompt = (
            "Rewrite the following text so it reads naturally, flows smoothly, "
            "and eliminates robotic AI patterns while keeping the exact core meaning intact:\n\n"
            f"{text}"
        )
        return self.client.generate(prompt, temperature=temperature)

    def polish(self, text: str, temperature: float = 0.7) -> str:
        """
        Second pass: Fine-tune tone and sentence rhythm.
        """
        prompt = (
            "Polish this text as an experienced human editor. Improve the rhythm, "
            "mix short and long sentences, and ensure it sounds completely authentic:\n\n"
            f"{text}"
        )
        return self.client.generate(prompt, temperature=temperature)

    def run(self, text: str) -> str:
        """
        Runs the full 2-step humanization pipeline using Gemini.
        """
        if not text or not text.strip():
            return ""

        # Step 1: Restructure
        first_pass = self.rewrite(text, temperature=0.9)
        
        # Step 2: Polish
        final_pass = self.polish(first_pass, temperature=0.7)

        return final_pass
