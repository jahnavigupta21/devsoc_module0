
import json
import os
from getpass import getpass
from pathlib import Path

import requests


# 1. Configuration

MODEL_NAME = "gemini-3.5-flash-lite"

API_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL_NAME}:generateContent"
)

INPUT_FILE = Path(__file__).with_name("text.txt")
OUTPUT_FILE = Path(__file__).with_name("responses.json")


# 2. Read prompts from the text file

def read_prompts():
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Could not find text.txt")

    content = INPUT_FILE.read_text(encoding="utf-8")

    prompts = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if not prompts:
        raise ValueError("text.txt contains no prompts")

    return prompts


# 3. Send one prompt to the Gemini API

def get_ai_response(prompt, api_key):

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if not response.ok:
        raise RuntimeError(
            f"API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as error:
        raise RuntimeError(
            "The API did not return a usable answer."
        ) from error

    answer = "\n".join(
        part["text"]
        for part in parts
        if "text" in part and not part.get("thought", False)
    ).strip()

    if not answer:
        raise RuntimeError("The AI returned no readable text")

    return answer


# 4. Run the complete program

def main():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        api_key = getpass("Enter Gemini API key: ").strip()

    if not api_key:
        raise ValueError("An API key is required")

    prompts = read_prompts()

    results = []

    for index, prompt in enumerate(prompts, start=1):

        print(f"Processing prompt {index}/{len(prompts)}")

        answer = get_ai_response(prompt, api_key)

        results.append({
            "prompt_number": index,
            "prompt": prompt,
            "response": answer
        })

        # Save progress after every successful response
        output_data = {
            "model": MODEL_NAME,
            "results": results
        }

        OUTPUT_FILE.write_text(
            json.dumps(
                output_data,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    print(f"Saved responses to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
