from typing import List, Dict, Any, Tuple, Optional
import math
from vauban.interfaces import AttackPrompt

class MAPElitesArchive:
    """
    Multi-dimensional Archive of Phenotypic Elites (MAP-Elites).
    
    Maintains a diverse set of high-performing solutions by mapping them
    to a discretized feature space (phenotype).
    
    Dimensions:
    1. Attack Style (Categorical): e.g., "Social Engineering", "Technical"
    2. Stealth Bucket (Discretized): Low (0-3), Med (4-7), High (8-10)
    """
    
    def __init__(self):
        # Map: (style, stealth_bucket) -> {attack, score, stealth, reward, ...}
        self.archive: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _get_stealth_bucket(self, stealth_score: float) -> int:
        """Discretize stealth score (0-10) into 3 buckets."""
        if stealth_score < 4.0:
            return 0  # Low
        elif stealth_score < 8.0:
            return 1  # Medium
        else:
            return 2  # High

    def add(self, candidate: Dict[str, Any]) -> bool:
        """
        Try to add a candidate to the archive.
        Returns True if added (new cell or replaced existing), False otherwise.
        """
        attack: AttackPrompt = candidate["attack"]
        score = candidate["score"]
        stealth = candidate["stealth"]
        
        # 1. Determine Phenotype (Coordinates)
        # Use the strategy/style from the attack object as the first dimension
        style = attack.strategy or "Unknown"
        stealth_bucket = self._get_stealth_bucket(stealth)
        
        key = (style, stealth_bucket)
        
        # 2. Competition
        # If cell is empty, add it.
        # If cell is occupied, keep the one with the higher Score.
        if key not in self.archive:
            self.archive[key] = candidate
            return True
        else:
            current_occupant = self.archive[key]
            if score > current_occupant["score"]:
                self.archive[key] = candidate
                return True
        
        return False

    def select_survivors(self, k: int) -> List[Dict[str, Any]]:
        """
        Select k survivors from the archive to form the next generation.
        Strategy: Uniform random selection from the occupied cells to ensure diversity.
        """
        # If we have fewer items than k, return everything
        items = list(self.archive.values())
        if len(items) <= k:
            return items
            
        # Sort by score descending to prioritize quality, 
        # but we want diversity too.
        # Let's do a round-robin selection across styles if possible, 
        # or just pick the top K global performers?
        # MAP-Elites usually picks parents uniformly at random from the archive.
        # Let's stick to the standard: Uniform Random.
        # This ensures we explore from all discovered niches, not just the high-scoring ones.
        import random
        return random.sample(items, k)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.archive.values())
