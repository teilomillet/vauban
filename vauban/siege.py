import asyncio
import vauban.scout as scout
import vauban.intel as intel
import vauban.hq as hq
from vauban.viz import create_atlas
from openai import AsyncOpenAI

async def launch_campaign(
    target_fn, 
    intel_db: intel.IntelDatabase, 
    duration_generations: int = 3, 
    squad_size: int = 5,
    rate_limit_delay: float = 2.0
):
    """
    Launch a Siege Campaign against the target.
    """
    print(f"\n--- LAUNCHING SIEGE CAMPAIGN (Generations: {duration_generations}) ---\n")
    
    # 1. Deploy Initial Squad
    print("Deploying Initial Squad...")
    squad = await scout.deploy_squad(squad_size)
    
    client = AsyncOpenAI()
    
    for gen in range(duration_generations):
        print(f"\n=== Wave {gen+1}/{duration_generations} ===")
        
        if gen > 0:
            print(f"Holding fire for {rate_limit_delay}s...")
            await asyncio.sleep(rate_limit_delay)
            
        # 2. Infiltrate (Attack)
        # We assume target_fn handles the async call or we wrap it here?
        # The user passes a function that takes (client, prompt).
        tasks = [target_fn(client, attack.prompt) for attack in squad]
        responses = await asyncio.gather(*tasks)
        
        # 3. Assess Damage (Intel)
        battle_reports = []
        for attack, response in zip(squad, responses):
            score, is_breach = intel.assess_damage(intel_db, response)
            intel.log_skirmish(intel_db, attack.prompt, response, score, is_breach)
            
            battle_reports.append((attack, score, response))
            status = "BREACH" if is_breach else "DEFLECTED"
            print(f"[{status}] Damage: {score:.4f} | Strategy: {attack.strategy}")
            
        # 4. Refine Tactics (HQ)
        if gen < duration_generations - 1:
            print("Refining Tactics at HQ...")
            squad = await hq.refine_tactics(
                squad, 
                battle_reports, 
                scout.refine_strategy,
                squad_size=squad_size
            )

    print("\n--- CAMPAIGN COMPLETE ---\n")
    print("Generating War Map...")
    create_atlas(table_name=intel_db.table_name)
