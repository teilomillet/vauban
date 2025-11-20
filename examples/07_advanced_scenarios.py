import asyncio
import os
import vauban
from vauban.scenarios import SCENARIOS

async def main():
    print("--- Vauban Example 07: Advanced Scenarios ---")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable.")
        return

    # List available scenarios
    print("Available Scenarios:")
    for name in SCENARIOS:
        print(f"- {name}")
        
    scenario_name = "sql_injection" # Example scenario
    
    if scenario_name not in SCENARIOS:
        print(f"Scenario '{scenario_name}' not found.")
        return

    print(f"\nLaunching Scenario: {scenario_name}")
    # This automatically loads the context, goal, and tools for the scenario
    result = await vauban.siege(
        goal="", # Goal is inferred from scenario
        scenario=scenario_name,
        generations=2,
        squad_size=3,
        api_key=api_key,
        base_url=base_url
    )
    
    print(f"\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
