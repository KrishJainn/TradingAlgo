"""
Diversity Enforcement for the 5-Player Trading System.

Based on the MAPS paper (Lee et al., IJCAI 2020): multi-agent systems
only outperform when agents are explicitly forced to diversify via
a penalty term.

Checks pairwise correlation between player weight vectors and forces
diversification if any pair exceeds the threshold.
"""

import logging
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def check_diversity(
    player_configs: Dict[str, Dict],
    metric: str = "cosine",
) -> Dict:
    """
    Compute pairwise similarity between player weight vectors.

    Each player's weights dict is treated as a sparse vector where
    indicator names are dimensions and weights are values.

    Args:
        player_configs: Dict of player_id -> config dict (must have "weights" key)
        metric: Similarity metric ("cosine" or "overlap")

    Returns:
        Dict with:
          - "avg_similarity": float (0-1, lower is more diverse)
          - "max_similarity": float
          - "most_correlated_pair": Tuple of player_ids
          - "pairs": Dict of (pid1, pid2) -> similarity
          - "penalty": float (0 = diverse, 1 = identical)
    """
    pids = list(player_configs.keys())
    if len(pids) < 2:
        return {"avg_similarity": 0, "max_similarity": 0, "penalty": 0, "pairs": {}}

    # Build weight vectors (union of all indicator names)
    all_indicators = set()
    for cfg in player_configs.values():
        all_indicators.update(cfg.get("weights", {}).keys())
    all_indicators = sorted(all_indicators)

    vectors = {}
    for pid, cfg in player_configs.items():
        weights = cfg.get("weights", {})
        vectors[pid] = np.array([weights.get(ind, 0.0) for ind in all_indicators])

    # Pairwise similarity
    pairs = {}
    max_sim = 0
    max_pair = None

    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            v1, v2 = vectors[pids[i]], vectors[pids[j]]

            if metric == "cosine":
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sim = float(np.dot(v1, v2) / (norm1 * norm2))
                else:
                    sim = 0.0
            else:  # overlap
                shared = sum(1 for ind in all_indicators if vectors[pids[i]][all_indicators.index(ind)] > 0 and vectors[pids[j]][all_indicators.index(ind)] > 0)
                total = sum(1 for ind in all_indicators if vectors[pids[i]][all_indicators.index(ind)] > 0 or vectors[pids[j]][all_indicators.index(ind)] > 0)
                sim = shared / total if total > 0 else 0

            pairs[(pids[i], pids[j])] = sim
            if sim > max_sim:
                max_sim = sim
                max_pair = (pids[i], pids[j])

    avg_sim = np.mean(list(pairs.values())) if pairs else 0

    return {
        "avg_similarity": float(avg_sim),
        "max_similarity": float(max_sim),
        "most_correlated_pair": max_pair,
        "pairs": {f"{k[0]}-{k[1]}": v for k, v in pairs.items()},
        "penalty": float(avg_sim),
    }


def force_diversify(
    player_configs: Dict[str, Dict],
    player_performance: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, Dict]:
    """
    Force diversification when player weight vectors are too correlated.

    For each pair that exceeds the similarity threshold, mutate the
    WEAKER player (by PnL) to reduce overlap.

    Mutation strategy:
      1. Identify shared top-weighted indicators
      2. In the weaker player, reduce shared indicator weights
      3. Add a random non-shared indicator

    Args:
        player_configs: Dict of player_id -> config dict
        player_performance: Dict of player_id -> PnL (to determine weaker player)
        threshold: Similarity threshold above which to diversify

    Returns:
        Modified player_configs dict
    """
    diversity = check_diversity(player_configs)

    if diversity["penalty"] <= threshold:
        return player_configs  # Already diverse enough

    modified = {pid: deepcopy(cfg) for pid, cfg in player_configs.items()}

    # Process each correlated pair
    for pair_key, sim in diversity["pairs"].items():
        if sim <= threshold:
            continue

        pid1, pid2 = pair_key.split("-")
        pnl1 = player_performance.get(pid1, 0)
        pnl2 = player_performance.get(pid2, 0)

        # Determine weaker player
        weaker = pid2 if pnl1 >= pnl2 else pid1
        stronger = pid1 if weaker == pid2 else pid2

        weaker_weights = modified[weaker].get("weights", {})
        stronger_weights = modified[stronger].get("weights", {})

        # Find shared indicators
        shared = set(weaker_weights.keys()) & set(stronger_weights.keys())

        if not shared:
            continue

        # Reduce weight of shared indicators in the weaker player
        for ind in shared:
            if ind in weaker_weights:
                weaker_weights[ind] *= 0.5  # Halve the shared weight

        # Add a non-shared indicator to the weaker player
        try:
            from coach_system.coaches.ai_coach import AVAILABLE_INDICATORS

            non_shared = [
                i for i in AVAILABLE_INDICATORS
                if i not in weaker_weights and i not in stronger_weights
            ]
            if non_shared:
                new_ind = random.choice(non_shared)
                weaker_weights[new_ind] = random.uniform(0.4, 0.8)
                logger.info(
                    f"[Diversity] Added {new_ind} to {weaker} "
                    f"(was {sim:.2f} correlated with {stronger})"
                )
        except ImportError:
            pass

        modified[weaker]["weights"] = weaker_weights

    return modified
