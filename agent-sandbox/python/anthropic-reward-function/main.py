"""
Buildfunctions + Anthropic: Code Generation with Reward Scoring

Generates a sorting function with Claude, then executes and scores it
in an isolated sandbox using a reward function that checks correctness
and edge case handling.
"""
import os
import re
import time
from pathlib import Path

import anthropic
import pytest
from dotenv import load_dotenv

from buildfunctions import Buildfunctions, CPUSandbox

load_dotenv()

API_TOKEN = os.environ.get("BUILDFUNCTIONS_API_TOKEN", "")
HANDLER_TEMPLATE = (Path(__file__).parent / "reward_handler.py").read_text()


def strip_markdown_fences(text: str) -> str:
    return re.sub(r"^```[\w]*\n|```$", "", text.strip(), flags=re.MULTILINE).strip()


@pytest.mark.asyncio
async def test_code_generation_with_reward():
    if not API_TOKEN:
        pytest.skip("Set BUILDFUNCTIONS_API_TOKEN in .env file")

    print("Testing Code Generation with Reward Scoring...\n")

    sandbox = None

    try:
        # Step 1: Authenticate
        print("1. Authenticating...")
        client = await Buildfunctions({"apiToken": API_TOKEN})
        print(f"   Authenticated as: {client.user.username}")

        # Step 2: Generate code with Claude
        print("\n2. Generating sorting function with Claude...")
        claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

        response = claude.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "Write a Python function called `sort_list` that takes a list and "
                    "returns it sorted. Handle edge cases like empty lists, single elements, "
                    "and mixed types. Return ONLY the code, no markdown, no explanations."
                ),
            }],
        )

        generated_code = strip_markdown_fences(response.content[0].text)
        print(f"   Generated code:\n{generated_code}\n")

        # Step 3: Create CPU Sandbox with reward function
        print("3. Creating CPU Sandbox with reward function...")
        handler_code = HANDLER_TEMPLATE.format(generated_code=generated_code)

        sandbox = await CPUSandbox.create({
            "name": f"reward-eval-{int(time.time())}",
            "language": "python",
            "code": handler_code,
            "memory": "512MB",
            "timeout": 30,
        })
        print(f"   CPU Sandbox created: {sandbox.name}")

        # Step 4: Run the reward function
        print("\n4. Running reward function...")
        result = await sandbox.run()
        print(f"   Result: {result.response}")

        # Step 5: Clean up
        print("\n5. Deleting CPU Sandbox...")
        await sandbox.delete()
        print("   CPU Sandbox deleted")

        print("\nCode generation with reward scoring test completed!")

    except Exception:
        if sandbox and sandbox.delete:
            try:
                await sandbox.delete()
            except Exception as e:
                print(f"Cleanup failed: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_code_generation_with_reward())
