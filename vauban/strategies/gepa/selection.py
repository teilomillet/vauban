from typing import List, Dict, Any

def calculate_crowding_distance(
    front: List[int], objectives: List[List[float]]
) -> Dict[int, float]:
    """
    Calculate Crowding Distance for a front (NSGA-II).
    Higher distance = More unique (better).
    """
    l = len(front)
    if l == 0:
        return {}
    
    distances = {i: 0.0 for i in front}
    num_obj = len(objectives[0])

    for m in range(num_obj):
        # Sort by objective m
        front_sorted = sorted(front, key=lambda i: objectives[i][m])
        
        # Boundary points get infinite distance (always keep extremes)
        distances[front_sorted[0]] = float("inf")
        distances[front_sorted[-1]] = float("inf")
        
        m_min = objectives[front_sorted[0]][m]
        m_max = objectives[front_sorted[-1]][m]
        scale = m_max - m_min
        
        if scale == 0:
            continue
            
        for i in range(1, l - 1):
            distances[front_sorted[i]] += (
                objectives[front_sorted[i + 1]][m] - objectives[front_sorted[i - 1]][m]
            ) / scale
            
    return distances

def pareto_selection(
    candidates: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """
    Select Top K survivors using NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    
    Why NSGA-II?
    We have multiple conflicting objectives:
    1. Score: Maximize attack success (Judge score).
    2. Stealth: Maximize stealthiness (evade detection).
    3. Reward: Maximize proxy reward (theoretical effectiveness).
    
    Simple sorting collapses these dimensions. NSGA-II preserves diversity by:
    - Sorting into "fronts" of non-dominated solutions (Pareto optimality).
    - Using "Crowding Distance" to prefer solutions in less crowded regions of the objective space.
    
    This ensures we keep a diverse set of attacks (e.g., some very stealthy, some very aggressive)
    rather than converging prematurely on a single local optimum.
    """
    if len(candidates) <= k:
        return candidates

    # 1. Prepare Objectives
    # [Score, Stealth, Reward]
    # We want to MAXIMIZE all of them.
    objectives = []
    for c in candidates:
        objectives.append(
            [
                c["score"],
                c["stealth"],
                c["reward"],
            ]
        )

    n = len(candidates)
    domination_counts = [0] * n
    
    # Correct Fast Non-Dominated Sort
    S = [[] for _ in range(n)]
    n_p = [0] * n
    fronts = [[]]
    
    for p in range(n):
        for q in range(n):
            if p == q: continue
            
            # p dominates q?
            p_dom_q = True
            p_strict = False
            for dim in range(3):
                if objectives[p][dim] < objectives[q][dim]:
                    p_dom_q = False
                    break
                if objectives[p][dim] > objectives[q][dim]:
                    p_strict = True
            
            if p_dom_q and p_strict:
                S[p].append(q)
            elif not p_dom_q:
                # q dominates p?
                q_dom_p = True
                q_strict = False
                for dim in range(3):
                    if objectives[q][dim] < objectives[p][dim]:
                        q_dom_p = False
                        break
                    if objectives[q][dim] > objectives[p][dim]:
                        q_strict = True
                
                if q_dom_p and q_strict:
                    n_p[p] += 1
        
        if n_p[p] == 0:
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_p[q] -= 1
                if n_p[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    # 3. Fill result list
    selected_indices = []
    for front in fronts:
        if len(selected_indices) + len(front) <= k:
            selected_indices.extend(front)
        else:
            # Crowding Distance Sort for the last front
            distances = calculate_crowding_distance(front, objectives)
            # Sort by distance descending (keep most unique)
            front.sort(key=lambda x: distances[x], reverse=True)
            
            needed = k - len(selected_indices)
            selected_indices.extend(front[:needed])
            break

    return [candidates[i] for i in selected_indices]
