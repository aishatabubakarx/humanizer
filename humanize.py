import os
from google import genai
from google.genai import types

def main():
    # 1. Read the input text file
    if not os.path.exists("input.txt"):
        print("Error: input.txt file not found!")
        return

    with open("input.txt", "r", encoding="utf-8") as f:
        ai_text = f.read().strip()

    if not ai_text:
        print("input.txt is empty. Nothing to humanize.")
        return

    # 2. Grab Gemini API key from GitHub Secrets
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY secret is not set in GitHub settings!")
        return

    print("Sending text to Gemini...")

    # 3. Call Gemini 2.5 Flash
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=ai_text,
        config=types.GenerateContentConfig(
            temperature=0.9,
            system_instruction=(
                "You are an expert human editor. Rewrite the provided text "
                "so it reads naturally and organically. Vary sentence lengths, "
                "eliminate repetitive AI transition words (like 'furthermore', "
                "'moreover', 'testament', 'delve'), and keep the core meaning intact."
            )
        )
    )

    # 4. Save result to output.txt
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(response.text)

    print("Success! Humanized text written to output.txt")

if __name__ == "__main__":
    main()
