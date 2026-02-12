"""
Probability of Backtest Overfitting (PBO).

Implements the Combinatorially Purged Cross-Validation (CPCV) framework
from Bailey et al. (2017) to estimate the probability that a backtested
strategy's performance is the result of overfitting rather than genuine
out-of-sample (OOS) alpha.

Algorithm
---------

1. Partition the *T* return observations into *S* contiguous, non-overlapping
   subsets (time blocks).

2. Enumerate all C(S, S/2) combinations that split the *S* subsets into
   a training half and a testing half.

3. For each combination *c*:

   a. Concatenate the training blocks to form the in-sample (IS) set.
   b. Concatenate the testing blocks to form the out-of-sample (OOS) set.
   c. Apply a purge of ``purge_bars`` observations at every boundary
      between IS and OOS blocks to prevent information leakage.
   d. Compute the Sharpe Ratio (or chosen performance metric) for every
      strategy in both IS and OOS.
   e. Identify the strategy with the highest IS Sharpe (the *selected*
      strategy).
   f. Record the OOS rank of this selected strategy among all strategies.

4. Compute the **logit** of the OOS relative rank:

       lambda_c = ln( rank_c / (1 - rank_c) )

   where rank_c in (0, 1) is the OOS rank normalised to the unit interval.

5. PBO = proportion of combinations where lambda_c < 0, i.e., where the
   IS-optimal strategy lands in the *worse* half of strategies OOS.

A PBO above 0.5 suggests the backtest is more likely overfit than not.

References
----------
Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017).
    "The Probability of Backtest Overfitting." *Journal of Computational
    Finance*, 20(4), 39--69.

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    Wiley, Chapters 11--12.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from risk_engine.config.constants import RISK_DEFAULTS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PBOResult:
    """Outcome of a Probability of Backtest Overfitting analysis.

    Attributes
    ----------
    pbo : float
        Probability of Backtest Overfitting in [0, 1].
        The fraction of CPCV combinations where the in-sample optimal
        strategy performed below the median out-of-sample.
        Values above 0.5 indicate likely overfitting.
    logit_distribution : np.ndarray
        1-D array of logit(rank) values for each CPCV combination.
        Negative values indicate the IS-best strategy ranked poorly OOS.
    n_combinations : int
        Total number of CPCV train/test combinations evaluated.
    is_overfit : bool
        ``True`` when ``pbo > 0.50`` --- the strategy is more likely
        overfit than not.
    """

    pbo: float
    logit_distribution: np.ndarray
    n_combinations: int
    is_overfit: bool


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════


class PBO:
    """Probability of Backtest Overfitting via CPCV.

    Estimates how likely a strategy selected by backtesting is the result
    of overfitting rather than genuine predictive ability.

    Parameters
    ----------
    n_splits : int
        Number of contiguous time blocks *S* to partition the data into.
        Must be even (so that S/2 blocks form each half).  Defaults to 10.
    purge_bars : int
        Number of return observations to purge at each IS/OOS block
        boundary to prevent information leakage from serial correlation.
        Defaults to ``RISK_DEFAULTS["meta_label_purge"]`` (5).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # 500 time steps, 8 strategies (columns)
    >>> returns = rng.normal(0.0, 0.01, size=(500, 8))
    >>> pbo = PBO(n_splits=6, purge_bars=3)
    >>> result = pbo.compute(returns)
    >>> result.pbo  # doctest: +SKIP
    0.65...
    >>> result.is_overfit  # doctest: +SKIP
    True
    """

    def __init__(
        self,
        n_splits: int = 10,
        purge_bars: int | None = None,
    ) -> None:
        if n_splits < 2 or n_splits % 2 != 0:
            raise ValueError(
                f"n_splits must be an even integer >= 2, got {n_splits}"
            )

        self.n_splits: int = n_splits
        self.purge_bars: int = (
            purge_bars if purge_bars is not None
            else int(RISK_DEFAULTS["meta_label_purge"])
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, returns_matrix: np.ndarray) -> PBOResult:
        """Run the full CPCV-based PBO analysis.

        Parameters
        ----------
        returns_matrix : np.ndarray
            2-D array of shape ``(T, N_strategies)`` where each column
            contains the return stream of a candidate strategy.

        Returns
        -------
        PBOResult
            Container with ``pbo``, ``logit_distribution``,
            ``n_combinations``, and ``is_overfit``.

        Raises
        ------
        ValueError
            If ``returns_matrix`` does not have at least 2 strategies
            (columns) or if T is too short to form the requested number
            of splits.

        Notes
        -----
        Computational cost is O(C(S, S/2) * N), where S = ``n_splits``
        and N = number of strategies.  For S = 10, C(10, 5) = 252, which
        is very tractable.  Avoid S > 16 unless performance is acceptable
        (C(16, 8) = 12 870).
        """
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)

        if returns_matrix.ndim == 1:
            raise ValueError(
                "returns_matrix must be 2-D with shape (T, N_strategies); "
                "got a 1-D array.  Need at least 2 strategies."
            )

        T, n_strategies = returns_matrix.shape

        if n_strategies < 2:
            raise ValueError(
                f"Need at least 2 strategies (columns), got {n_strategies}"
            )

        min_rows: int = self.n_splits * max(2, self.purge_bars + 1)
        if T < min_rows:
            raise ValueError(
                f"returns_matrix has {T} rows but at least {min_rows} are "
                f"required for {self.n_splits} splits with "
                f"purge_bars={self.purge_bars}."
            )

        # ---- Generate CPCV combinations ----
        cpcv_combos: list[tuple[list[int], list[int]]] = (
            self._generate_cpcv_combinations(self.n_splits)
        )
        n_combos: int = len(cpcv_combos)

        # ---- Block boundaries ----
        block_edges: np.ndarray = np.linspace(0, T, self.n_splits + 1, dtype=int)

        logits: list[float] = []

        for train_block_ids, test_block_ids in cpcv_combos:
            # Build row indices for train / test from block IDs
            train_rows: list[int] = []
            for bid in train_block_ids:
                train_rows.extend(range(int(block_edges[bid]), int(block_edges[bid + 1])))

            test_rows: list[int] = []
            for bid in test_block_ids:
                test_rows.extend(range(int(block_edges[bid]), int(block_edges[bid + 1])))

            # Purge
            train_data, test_data = self._purged_split(
                returns_matrix, train_rows, test_rows
            )

            # Skip if purging left too few rows
            if train_data.shape[0] < 2 or test_data.shape[0] < 2:
                continue

            # ---- Compute Sharpe for each strategy IS and OOS ----
            is_sharpes: np.ndarray = self._column_sharpes(train_data)
            oos_sharpes: np.ndarray = self._column_sharpes(test_data)

            # ---- Identify IS-best strategy ----
            best_is_idx: int = int(np.argmax(is_sharpes))

            # ---- Rank the IS-best strategy's OOS Sharpe ----
            # Rank among all strategies: 1 = worst, N = best
            oos_sr_of_best: float = oos_sharpes[best_is_idx]
            rank: int = int(np.sum(oos_sharpes <= oos_sr_of_best))

            # Normalise rank to (0, 1) --- avoid exact 0 or 1 for logit
            rank_norm: float = rank / (n_strategies + 1.0)

            # Logit transformation: ln(rank / (1 - rank))
            logit_val: float = math.log(rank_norm / (1.0 - rank_norm))
            logits.append(logit_val)

        logit_arr: np.ndarray = np.array(logits, dtype=np.float64)

        # PBO = fraction of combos where logit < 0
        if len(logit_arr) > 0:
            pbo_val: float = float(np.mean(logit_arr < 0.0))
        else:
            pbo_val = 1.0  # No valid combos -> assume overfit

        is_overfit: bool = pbo_val > 0.50

        logger.debug(
            "PBO=%.4f | combos=%d | logit_mean=%.4f | overfit=%s",
            pbo_val,
            len(logit_arr),
            float(np.mean(logit_arr)) if len(logit_arr) > 0 else float("nan"),
            is_overfit,
        )

        return PBOResult(
            pbo=round(pbo_val, 6),
            logit_distribution=logit_arr,
            n_combinations=len(logit_arr),
            is_overfit=is_overfit,
        )

    # ------------------------------------------------------------------
    # CPCV combination generator
    # ------------------------------------------------------------------

    def _generate_cpcv_combinations(
        self, n_splits: int
    ) -> list[tuple[list[int], list[int]]]:
        """Generate all C(S, S/2) train/test block-index combinations.

        Each combination assigns exactly S/2 of the S contiguous time
        blocks to the training set and the remaining S/2 to the test set.

        Parameters
        ----------
        n_splits : int
            Total number of time blocks *S*.  Must be even.

        Returns
        -------
        list[tuple[list[int], list[int]]]
            Each element is ``(train_block_ids, test_block_ids)`` where
            block IDs are integers in ``range(n_splits)``.
        """
        half: int = n_splits // 2
        all_ids: list[int] = list(range(n_splits))

        combos: list[tuple[list[int], list[int]]] = []
        for test_ids_tuple in combinations(all_ids, half):
            test_ids: list[int] = list(test_ids_tuple)
            train_ids: list[int] = [i for i in all_ids if i not in test_ids_tuple]
            combos.append((train_ids, test_ids))

        return combos

    # ------------------------------------------------------------------
    # Purged split
    # ------------------------------------------------------------------

    def _purged_split(
        self,
        data: np.ndarray,
        train_idx: list[int],
        test_idx: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply temporal purging to train/test row indices.

        Removes ``purge_bars`` observations from the training set that
        fall within the embargo zone around each contiguous test block
        boundary.  This prevents information leakage caused by serial
        correlation in financial returns.

        Parameters
        ----------
        data : np.ndarray
            Full returns matrix of shape ``(T, N_strategies)``.
        train_idx : list[int]
            Row indices assigned to the training set.
        test_idx : list[int]
            Row indices assigned to the test set.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(train_data, test_data)`` --- sliced views of ``data``
            with the purge applied to the training set.

        Notes
        -----
        The embargo zone extends ``purge_bars`` rows *before* the start
        and *after* the end of each contiguous test segment.  Any training
        row inside an embargo zone is dropped.

        Reference: Lopez de Prado (2018), Chapter 7.
        """
        if self.purge_bars <= 0:
            return data[train_idx], data[test_idx]

        test_set: set[int] = set(test_idx)

        # Identify contiguous test segments and build embargo zones
        sorted_test: list[int] = sorted(test_idx)
        embargo: set[int] = set()

        # Walk through sorted test indices and find segment boundaries
        if len(sorted_test) > 0:
            seg_start: int = sorted_test[0]
            seg_end: int = sorted_test[0]

            for i in range(1, len(sorted_test)):
                if sorted_test[i] == seg_end + 1:
                    seg_end = sorted_test[i]
                else:
                    # End of a contiguous segment --- apply purge
                    self._add_embargo(embargo, seg_start, seg_end, len(data))
                    seg_start = sorted_test[i]
                    seg_end = sorted_test[i]

            # Final segment
            self._add_embargo(embargo, seg_start, seg_end, len(data))

        # Remove embargoed rows from training set
        purged_train: list[int] = [
            i for i in train_idx
            if i not in embargo and i not in test_set
        ]

        return data[purged_train], data[test_idx]

    def _add_embargo(
        self,
        embargo: set[int],
        seg_start: int,
        seg_end: int,
        T: int,
    ) -> None:
        """Add embargo zone rows around a contiguous test segment.

        Parameters
        ----------
        embargo : set[int]
            Mutable set to which embargo row indices are added.
        seg_start : int
            First row index of the contiguous test segment.
        seg_end : int
            Last row index of the contiguous test segment.
        T : int
            Total number of rows in the dataset (for boundary clipping).
        """
        for j in range(
            max(0, seg_start - self.purge_bars),
            min(T, seg_end + self.purge_bars + 1),
        ):
            embargo.add(j)

    # ------------------------------------------------------------------
    # Helper: column-wise Sharpe ratios
    # ------------------------------------------------------------------

    @staticmethod
    def _column_sharpes(data: np.ndarray) -> np.ndarray:
        """Compute the Sharpe Ratio for each column (strategy) of a matrix.

        Sharpe is computed as ``mean / std`` (excess return over zero,
        which is standard in relative strategy comparison within CPCV).
        Strategies with near-zero standard deviation receive a Sharpe
        of 0.0 to avoid division-by-zero artefacts.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape ``(T, N_strategies)``.

        Returns
        -------
        np.ndarray
            1-D array of Sharpe Ratios, one per column.
        """
        means: np.ndarray = np.mean(data, axis=0)
        stds: np.ndarray = np.std(data, axis=0, ddof=1)

        # Protect against near-zero std
        safe_stds: np.ndarray = np.where(stds > 1.0e-15, stds, 1.0)
        sharpes: np.ndarray = np.where(stds > 1.0e-15, means / safe_stds, 0.0)

        return sharpes
