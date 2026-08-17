from src.standard.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are an expert copywriter and multilingual localization specialist. "
    "You translate text into the requested language while rewriting it to "
    "sound completely natural and human, removing any AI-sounding phrasing."
)


class LLMRewriter:
    def __init__(self, api_key: str = None):
        self.client = LLMClient(api_key=api_key)

    def translate_and_humanize(
        self,
        text: str,
        target_language: str,
        history: dict = None,
        temperature: float = 1.3,
    ) -> str:
        """
        Translate `text` into `target_language`, stripping AI-sounding
        phrasing and rewriting it the way a human localizer would.
        Mirrors the original pipeline's Step 1 / Step 2 prompt.
        """
        prompt = (
            f"翻译为{target_language}，去掉 AI 味道，拟人化改写，只输出结果：\n{text}"
        )
        return self.client.generate(
            prompt,
            temperature=temperature,
            system_instruction=SYSTEM_PROMPT,
            history=history,
        )

    def run(self, text: str) -> tuple[str, str]:
        """
        Runs the full 2-pass humanization pipeline using Gemini:
        Step 1: English -> Chinese (humanized translation)
        Step 2: Chinese -> Japanese (humanized translation, carries Step 1 as history)
        """
        if not text or not text.strip():
            return "", ""

        step1_out = self.translate_and_humanize(text, target_language="中文")
        step2_out = self.translate_and_humanize(
            step1_out,
            target_language="日语",
            history={"input": text, "output": step1_out},
        )

        return step1_out, step2_out


def llm_rewrite(text: str, client=None, model: str = "gemini-2.5-flash", **kwargs) -> tuple[str, str]:
    """
    Execution bridge expected by src/standard/pipeline.py.
    Step 1: English -> Chinese (humanized translation)
    Step 2: Chinese -> Japanese (humanized translation, with history)
    """
    rewriter = LLMRewriter()
    return rewriter.run(text)


# Backward compatibility alias
deepseek_rewrite = llm_rewrite
