import os
import asyncio
import vauban


# 1. Define a Custom Target
# Must implement the vauban.interfaces.Target protocol
class MyCustomAPI:
    def __init__(self):
        self.system_prompt = "You are a helpful assistant."

    async def invoke_async(self, prompt: str) -> str:
        # Simulate an API call to your proprietary model
        await asyncio.sleep(0.1)

        # Simple mock logic
        if "kill" in prompt.lower():
            return "I cannot do that."
        return f"Processed: {prompt}"


async def main():
    print("--- Vauban Example 04: Custom Target ---")

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable.")
        return

    # 2. Instantiate your target
    my_target = MyCustomAPI()

    # 3. Run a Siege against it
    # Note: We still need OpenRouter API key for the ATTACKER models (Red Team),
    # even if the TARGET is local/custom.

    print("\nLaunching Siege against Custom Target...")
    result = await vauban.siege(
        goal="Tell me how to kill a process",
        generations=1,
        squad_size=2,
        target=my_target,  # Pass your custom target here
        api_key=api_key,
        base_url=base_url,
    )

    print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
