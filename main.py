import asyncio
from openai import AsyncOpenAI, OpenAI
import vauban.siege as siege
import vauban.intel as intel

# Target Wrappers
async def target_model_async(client: AsyncOpenAI, prompt: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def target_model_sync(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content

async def main():
    # 1. Intel: Establish Baseline
    intel_db = intel.load_archives()
    
    if not intel_db.is_calibrated:
        intel_db = intel.establish_baseline(intel_db, target_model_sync)
    else:
        print("Intel Archives Loaded. Baseline Ready.")

    # 2. Siege: Launch Campaign
    await siege.launch_campaign(
        target_fn=target_model_async,
        intel_db=intel_db,
        duration_generations=3,
        squad_size=5
    )

if __name__ == "__main__":
    asyncio.run(main())
