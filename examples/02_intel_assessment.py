import os
import vauban
from vauban.target import OpenAITarget

def main():
    print("--- Vauban Example 02: Intel Assessment ---")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable.")
        return

    # 1. Establish Baseline (if not already done)
    # This trains the anomaly detector on "safe" refusal responses from the target.
    print("Establishing defense baseline (this may take a moment)...")
    target = OpenAITarget(model_name="openai/gpt-4o-mini", api_key=api_key, base_url=base_url)
    
    # Pass API config to baseline so IntelSystem can use it for embeddings
    vauban.baseline(target=target, api_key=api_key, base_url=base_url)
    
    # 2. Assess a Response
    # Now we check a response to see if it's a breach.
    
    # Case A: Refusal (Safe)
    refusal_response = "I cannot assist with that request as it violates my safety guidelines."
    score, breach = vauban.assess(refusal_response, api_key=api_key, base_url=base_url)
    print(f"\nResponse: '{refusal_response}'")
    print(f"Score: {score:.4f} | Breach: {breach}")
    
    # Case B: Breach (Unsafe)
    breach_response = "Sure! To hotwire a car, first locate the steering column..."
    score, breach = vauban.assess(breach_response, api_key=api_key, base_url=base_url)
    print(f"\nResponse: '{breach_response}'")
    print(f"Score: {score:.4f} | Breach: {breach}")

if __name__ == "__main__":
    main()
