"""
Fitness Calculation module.

Calculates fitness scores for Super Indicator DNA configurations.
Fitness = f(Net Profit, Sharpe Ratio, Max Drawdown, Consistency)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitnessWeights:
    """Weights for fitness function components."""
    net_profit: float = 0.4
    sharpe_ratio: float = 0.3
    max_drawdown: float = 0.2
    consistency: float = 0.1


@dataclass
class FitnessResult:
    """Complete fitness evaluation result."""
    fitness_score: float
    profit_score: float
    sharpe_score: float
    drawdown_score: float
    consistency_score: float
    penalties_applied: List[str]
    is_valid: bool
    reason: str = ""


class FitnessCalculator:
    """
    Calculates fitness scores for evolved strategies.

    Fitness Function:
    F = (Profit Score × 0.4) + (Sharpe Score × 0.3) +
        (Drawdown Score × 0.2) + (Consistency Score × 0.1)

    With penalties for:
    - Insufficient trades (< 30)
    - Very low win rate (< 30%)
    - Large drawdown (> 25%)
    - Negative Sharpe ratio
    """

    def __init__(self,
                 weights: FitnessWeights = None,
                 min_trades: int = 30,
                 min_win_rate: float = 0.3,
                 max_acceptable_drawdown: float = 0.25):
        """
        Initialize fitness calculator.

        Args:
            weights: Component weights for fitness function
            min_trades: Minimum trades for valid fitness
            min_win_rate: Minimum win rate before penalty
            max_acceptable_drawdown: Maximum drawdown before penalty
        """
        self.weights = weights or FitnessWeights()
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.max_acceptable_drawdown = max_acceptable_drawdown

    def calculate_fitness(self, metrics: Dict) -> FitnessResult:
        """
        Calculate fitness score from performance metrics.

        Args:
            metrics: Dict with keys: net_profit, sharpe_ratio, max_drawdown,
                     win_rate, total_trades, profit_factor, etc.

        Returns:
            FitnessResult with detailed breakdown
        """
        penalties = []

        # Extract metrics
        net_profit = metrics.get('net_profit', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        total_trades = metrics.get('total_trades', 0)
        profit_factor = metrics.get('profit_factor', 0)

        # Check validity
        if total_trades < self.min_trades:
            return FitnessResult(
                fitness_score=0.0,
                profit_score=0.0,
                sharpe_score=0.0,
                drawdown_score=0.0,
                consistency_score=0.0,
                penalties_applied=[f"Insufficient trades: {total_trades}"],
                is_valid=False,
                reason=f"Need at least {self.min_trades} trades"
            )

        # 1. Profit Score (sigmoid normalization centered at $10,000)
        profit_score = self._score_profit(net_profit)

        # 2. Sharpe Score (target: 1.5+)
        sharpe_score = self._score_sharpe(sharpe)

        # 3. Drawdown Score (lower is better)
        drawdown_score = self._score_drawdown(max_dd)

        # 4. Consistency Score (based on profit factor and win rate)
        consistency_score = self._score_consistency(win_rate, profit_factor)

        # Combine scores
        fitness = (
            profit_score * self.weights.net_profit +
            sharpe_score * self.weights.sharpe_ratio +
            drawdown_score * self.weights.max_drawdown +
            consistency_score * self.weights.consistency
        )

        # Apply penalties
        if win_rate < self.min_win_rate:
            fitness *= 0.5  # Harsh penalty for low win rate
            penalties.append(f"Low win rate: {win_rate:.1%}")

        if max_dd > self.max_acceptable_drawdown:
            fitness *= 0.6
            penalties.append(f"High drawdown: {max_dd:.1%}")

        if sharpe < 0:
            fitness *= 0.4  # Harsh penalty for negative Sharpe
            penalties.append(f"Negative Sharpe: {sharpe:.2f}")

        if total_trades < 50:
            fitness *= 0.7
            penalties.append(f"Low trade count: {total_trades}")

        # BONUS MULTIPLIERS for exceptional performance
        if win_rate >= 0.70:  # 70%+ win rate
            fitness *= 1.3
            penalties.append(f"BONUS: Excellent win rate {win_rate:.1%}")

        if win_rate >= 0.65 and sharpe >= 5.0:  # Great combo
            fitness *= 1.2
            penalties.append(f"BONUS: High Sharpe + Win rate combo")

        if net_profit >= 100000 and win_rate >= 0.60:  # $100k+ with 60%+ WR
            fitness *= 1.25
            penalties.append(f"BONUS: High profit with good consistency")

        if profit_factor >= 2.0:  # Profit factor 2:1 or better
            fitness *= 1.15
            penalties.append(f"BONUS: Excellent profit factor {profit_factor:.2f}")

        return FitnessResult(
            fitness_score=max(0, fitness),
            profit_score=profit_score,
            sharpe_score=sharpe_score,
            drawdown_score=drawdown_score,
            consistency_score=consistency_score,
            penalties_applied=penalties,
            is_valid=True
        )

    def _score_profit(self, net_profit: float) -> float:
        """
        Score net profit using sigmoid normalization.

        Maps profit to (0, 1) with inflection at $10,000.
        """
        # Sigmoid: 1 / (1 + e^(-x/k))
        # k = 10000 means $10k profit -> 0.73 score
        return 1 / (1 + np.exp(-net_profit / 10000))

    def _score_sharpe(self, sharpe: float) -> float:
        """
        Score Sharpe ratio.

        Target Sharpe of 1.5 maps to ~0.73.
        """
        if sharpe < 0:
            return 0.0

        # Sigmoid with inflection at 1.0
        return 1 / (1 + np.exp(-(sharpe - 1.0) * 2))

    def _score_drawdown(self, max_dd: float) -> float:
        """
        Score max drawdown (lower is better).

        Linear penalty: 30% drawdown = 0.7 score.
        """
        return max(0, 1 - max_dd)

    def _score_consistency(self, win_rate: float, profit_factor: float) -> float:
        """
        Score consistency based on win rate and profit factor.

        High consistency = steady profits without large swings.
        HEAVILY weighted towards win rate for predictable trading.
        """
        # Win rate contribution (70% weight) - HEAVILY WEIGHTED
        # 55% = 0.7, 60% = 0.85, 65% = 0.93, 70% = 1.0
        if win_rate >= 0.70:
            wr_score = 1.0
        elif win_rate >= 0.65:
            wr_score = 0.9 + (win_rate - 0.65) * 2  # 0.9 to 1.0
        elif win_rate >= 0.55:
            wr_score = 0.7 + (win_rate - 0.55) * 2  # 0.7 to 0.9
        else:
            wr_score = max(0, win_rate / 0.55 * 0.7)  # 0 to 0.7

        # Profit factor contribution (30% weight)
        # PF of 1.5 = 0.75 score, PF of 2.0 = 1.0 score
        pf_score = min(1.0, profit_factor / 2.0) if profit_factor > 0 else 0

        return (wr_score * 0.7) + (pf_score * 0.3)

    def compare(self, metrics1: Dict, metrics2: Dict) -> int:
        """
        Compare two strategies.

        Returns:
            1 if metrics1 is better, -1 if metrics2 is better, 0 if equal
        """
        f1 = self.calculate_fitness(metrics1)
        f2 = self.calculate_fitness(metrics2)

        if f1.fitness_score > f2.fitness_score:
            return 1
        elif f2.fitness_score > f1.fitness_score:
            return -1
        return 0


def calculate_metrics_from_trades(trades: List[Dict]) -> Dict:
    """
    Calculate performance metrics from a list of trades.

    Args:
        trades: List of trade dictionaries with 'net_pnl', 'net_pnl_pct', etc.

    Returns:
        Dict with calculated metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'net_profit': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }

    # Extract returns
    pnls = [t.get('net_pnl', 0) for t in trades]
    pnl_pcts = [t.get('net_pnl_pct', 0) for t in trades]

    # Win rate
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / len(trades) if trades else 0

    # Net profit
    net_profit = sum(pnls)

    # Gross profit/loss
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    if profit_factor == float('inf'):
        profit_factor = 10.0  # Cap at reasonable value

    # Sharpe ratio (annualized)
    if len(pnl_pcts) > 1:
        avg_return = np.mean(pnl_pcts)
        std_return = np.std(pnl_pcts)
        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / (np.abs(running_max) + 1e-10)
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Long/Short breakdown
    longs = [t for t in trades if t.get('direction') == 'LONG']
    shorts = [t for t in trades if t.get('direction') == 'SHORT']
    long_winners = [t for t in longs if t.get('net_pnl', 0) > 0]
    short_winners = [t for t in shorts if t.get('net_pnl', 0) > 0]

    return {
        'total_trades': len(trades),
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'win_rate': win_rate,
        'long_win_rate': len(long_winners) / len(longs) if longs else 0,
        'short_win_rate': len(short_winners) / len(shorts) if shorts else 0,
        'net_profit': net_profit,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'avg_trade': np.mean(pnls) if pnls else 0,
        'avg_winner': np.mean([p for p in pnls if p > 0]) if winners else 0,
        'avg_loser': np.mean([p for p in pnls if p <= 0]) if losers else 0
    }
