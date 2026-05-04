"""This module contains the Acquirer class, which is used to gather inputs for
a subsequent round of exploration based on prior prediction data."""
import heapq
from itertools import chain
import math
from timeit import default_timer
from typing import Dict, Iterable, List, Mapping, Optional, Set, TypeVar, Union

import numpy as np
from tqdm import tqdm

from molpal.acquirer import metrics

T = TypeVar("T")


class Acquirer:
    """An Acquirer acquires inputs from an input pool for exploration.

    Attributes
    ----------
    size : int
        the size of the pool this acquirer will work on
    metric : str (Default = 'greedy')
        the alias of the metric to use
    epsilon : float
        the fraction of each batch that should be acquired randomly
    temp_i : Optional[float]
        the initial temperature value to use for clustered acquisition
    temp_f : Optional[float]
        the final temperature value to use "..."
    xi: float
        the xi value for EI and PI metrics
    beta : float
        the beta value for the UCB metric
    stochastic_preds : bool
        whether the prediction values are generated through stochastic means
    threshold : float
        the threshold value to use in the random_threshold metric
    verbose : int
        the level of output the acquirer should print

    Parameters
    ----------
    size : int
    init_size : Union[int, float] (Default = 0.01)
        the number of ligands or fraction of the pool to acquire initially.
    batch_sizes : Iterable[Union[int, float]] (Default = [0.01])
        the number of inputs or fraction of the pool to acquire in each
        successive batch. Will successively use each value in the list after
        each call to acquire_batch(), repeating the final value as necessary.
    metric : str (Default = 'greedy')
    epsilon : float (Default = 0.)
    temp_i : Optional[float] (Default = None)
    temp_f : Optional[float] (Default = 1.)
    xi: float (Default = 0.01)
    beta : int (Default = 2)
    threshold : float (Default = float('-inf'))
    seed : Optional[int] (Default = None)
        the random seed to use for initial batch acquisition
    verbose : int (Default = 0)
    **kwargs
        additional and unused keyword arguments
    """

    def __init__(
        self,
        size: int,
        init_size: Union[int, float] = 0.01,
        batch_sizes: Iterable[Union[int, float]] = [0.01],
        metric: str = "greedy",
        epsilon: float = 0.0,
        beta: int = 2,
        xi: float = 0.01,
        threshold: float = float("-inf"),
        diversity_alpha: float = 0.8,
        temp_i: Optional[float] = None,
        temp_f: Optional[float] = 1.0,
        seed: Optional[int] = None,
        verbose: int = 0,
        **kwargs,
    ):
        self.size = size
        self.init_size = init_size
        self.batch_sizes = batch_sizes

        self.metric = metric
        self.stochastic_preds = False

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"Epsilon(={epsilon}) must be in [0, 1]")
        self.epsilon = epsilon

        self.beta = beta
        self.xi = xi
        self.threshold = threshold

        # MD-EI parameter: balance between exploitation and exploration
        if not 0.0 <= diversity_alpha <= 1.0:
            raise ValueError(f"diversity_alpha(={diversity_alpha}) must be in [0, 1]")
        self.diversity_alpha = diversity_alpha

        self.temp_i = temp_i
        self.temp_f = temp_f
        self.seed = seed
        self.verbose = verbose

        metrics.set_seed(self.seed)

    def __len__(self) -> int:
        return self.size

    @property
    def needs(self) -> Set[str]:
        """the set of values this acquirer needs to calculate acquisition utilities"""
        return metrics.get_needs(self.metric)

    @property
    def init_size(self) -> int:
        """the number of inputs to acquire initially"""
        return self.__init_size

    @init_size.setter
    def init_size(self, init_size: Union[int, float]):
        if isinstance(init_size, float):
            if init_size < 0 or init_size > 1:
                raise ValueError(f"init_size(={init_size} must be in [0, 1]")
            init_size = math.ceil(self.size * init_size)
        if init_size < 0:
            raise ValueError(f"init_size(={init_size}) must be positive")

        self.__init_size = init_size

    @property
    def batch_sizes(self) -> List[int]:
        """the number of inputs to acquire in exploration batch"""
        return self.__batch_sizes

    @batch_sizes.setter
    def batch_sizes(self, batch_sizes: Iterable[Union[int, float]]):
        self.__batch_sizes = [bs for bs in batch_sizes]

        for i, bs in enumerate(self.__batch_sizes):
            if isinstance(bs, float):
                if bs < 0 or bs > 1:
                    raise ValueError(f"batch_size(={bs} must be in [0, 1]")
                self.__batch_sizes[i] = math.ceil(self.size * bs)
            if bs < 0:
                raise ValueError(f"batch_size(={bs} must be positive")

    def batch_size(self, t: int) -> int:
        try:
            batch_size = self.batch_sizes[t]
        except (IndexError, TypeError):
            batch_size = self.batch_sizes[-1]

        return batch_size

    def reset(self):
        """reset the random state of the metrics module"""
        metrics.set_seed(self.seed)

    def acquire_initial(
        self,
        xs: Iterable[T],
        cluster_ids: Optional[Iterable[int]] = None,
        cluster_sizes: Optional[Mapping[int, int]] = None,
    ) -> List[T]:
        """Acquire an initial set of inputs to explore

        Parameters
        ----------
        xs : Iterable[T]
            an iterable of the inputs to acquire
        cluster_ids : Optional[Iterable[int]] (Default = None)
            a parallel iterable for the cluster ID of each input
        cluster_sizes : Optional[Mapping[int, int]] (Default = None)
            a mapping from a cluster id to the sizes of that cluster

        Returns
        -------
        List[T]
            the list of inputs to explore
        """
        U = metrics.random(np.empty(self.size))

        if cluster_ids is None and cluster_sizes is None:
            heap = []
            for x, u in tqdm(zip(xs, U), total=U.size, desc="Acquiring"):
                if len(heap) < self.init_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))
        else:
            d_cid_heap = {
                cid: ([], math.ceil(self.init_size * cluster_size / U.size))
                for cid, cluster_size in cluster_sizes.items()
            }

            for x, u, cid in tqdm(
                zip(xs, U, cluster_ids), "Acquiring", U.size, disable=self.verbose < 1
            ):
                heap, heap_size = d_cid_heap[cid]
                if len(heap) < heap_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))

            heaps = [heap for heap, _ in d_cid_heap.values()]
            heap = list(chain(*heaps))

        if self.verbose > 0:
            print(f"  Selected {len(heap)} initial samples")

        return [x for _, x in heap]

    def acquire_batch(
        self,
        xs: Iterable[T],
        y_means: Iterable[float],
        y_vars: Iterable[float],
        explored: Optional[Mapping] = None,
        k: int = 1,
        cluster_ids: Optional[Iterable[int]] = None,
        cluster_sizes: Optional[Mapping[int, int]] = None,
        t: Optional[int] = None,
        fingerprints: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[T]:
        """Acquire a batch of inputs to explore

        Parameters
        ----------
        xs : Iterable[T]
            an iterable of the inputs to acquire
        y_means : Iterable[float]
            the predicted input values
        y_vars : Iterable[float]
            the variances of the predicted input values
        explored : Mapping[T, float] (Default = {})
            the set of explored inputs and their associated scores
        k : int, default=1
            the number of top-scoring compounds we are searching for. By
            default, assume we're looking for only the top-1 compound
        cluster_ids : Optional[Iterable[int]] (Default = None)
            a parallel iterable for the cluster ID of each input
        cluster_sizes : Optional[Mapping[int, int]] (Default = None)
            a mapping from a cluster id to the sizes of that cluster
        t : Optional[int] (Default = None)
            the current iteration of batch acquisition
        fingerprints : Optional[np.ndarray] (Default = None)
            molecular fingerprints for diversity calculation (used by MD-EI)
        is_random : bool (Default = False)
            are the y_means generated through stochastic methods?

        Returns
        -------
        List[T]
            a list of selected inputs in descending order of acquisition utility
        """
        if explored:
            ys = list(explored.values())
            Y = np.nan_to_num(np.array(ys, float), nan=-np.inf)
            current_max = np.partition(Y, -k)[-k] if len(Y) >= k else Y.max()
        else:
            explored = {}
            current_max = float("-inf")

        batch_size = self.batch_size(t)

        begin = default_timer()

        Y_mean = np.array(y_means)
        Y_var = np.array(y_vars)

        # MD-EI requires iterative selection
        if self.metric == "md_ei" and fingerprints is not None:
            return self._acquire_batch_md_ei(
                xs, Y_mean, Y_var, current_max, batch_size, explored, fingerprints
            )

        # Standard acquisition for other metrics
        if self.verbose > 1:
            print("Calculating acquisition utilities ...", end=" ")

        U = metrics.calc(
            self.metric,
            Y_mean,
            Y_var,
            current_max,
            self.threshold,
            self.beta,
            self.xi,
            self.stochastic_preds,
        )

        idxs = np.random.choice(U.size, math.ceil(batch_size * self.epsilon), False)
        U[idxs] = np.inf

        if self.verbose > 1:
            print("Done!")
        if self.verbose > 2:
            total = default_timer() - begin
            mins, secs = divmod(int(total), 60)
            print(f"      Utility calculation took {mins}m {secs}s")

        if cluster_ids is None and cluster_sizes is None:
            heap = []
            for x, u in tqdm(zip(xs, U), "Acquiring", U.size, disable=self.verbose < 1):
                if x in explored:
                    continue

                if len(heap) < batch_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))
        else:
            raise NotImplementedError
            # NOTE(degraff): this is broken for epsilon approaches
            #   the random indices are not distributed evenly amongst clusters
            # NOTE(degraff): this is also broken for pool exhaustion logic in the explorer

            # d_cid_heap = {
            #     cid: ([], math.ceil(batch_size * cluster_size / U.size))
            #     for cid, cluster_size in cluster_sizes.items()
            # }

            # global_pred_max = float("-inf")

            # for x, y_pred, u, cid in tqdm(
            #     zip(xs, Y_mean, U, cluster_ids), total=U.size, desc="Acquiring"
            # ):
            #     global_pred_max = max(y_pred, global_pred_max)

            #     if x in explored:
            #         continue

            #     heap, heap_size = d_cid_heap[cid]
            #     if len(heap) < heap_size:
            #         heapq.heappush(heap, (u, x))
            #     else:
            #         heapq.heappushpop(heap, (u, x))

            # if self.temp_i and self.temp_f:
            #     d_cid_heap = self.scale_heaps(d_cid_heap, global_pred_max, t)

            # heaps = [heap for heap, _ in d_cid_heap.values()]
            # heap = list(chain(*heaps))

        if self.verbose > 1:
            print(f"Selected {len(heap)} new samples")
        if self.verbose > 2:
            total = default_timer() - begin
            mins, secs = divmod(int(total), 60)
            print(f"      Batch acquisition took {mins}m {secs}s")

        return [x for _, x in sorted(heap, reverse=True)]

    def _acquire_batch_md_ei(
        self,
        xs: Iterable[T],
        Y_mean: np.ndarray,
        Y_var: np.ndarray,
        current_max: float,
        batch_size: int,
        explored: Mapping,
        fingerprints: np.ndarray,
    ) -> List[T]:
        """Acquire a batch using MD-EI v2 with iterative MaxMin selection.

        Improvements over the original MD-EI:
        1. MaxMin diversity (min distance to selected set) instead of
        average distance — much more discriminative.
        2. Training-set-aware: the already-explored molecules are treated
        as part of the "selected set" from the very first pick, so the
        acquirer avoids regions the model has already saturated.
        3. Rank-based EI normalization instead of max-normalization, so
        the EI signal stays well-distributed across the candidate pool
        and doesn't get crushed by the long-tailed EI distribution.
        4. Running min-distance array updated in O(n) per iteration.
        """
        begin = default_timer()

        if self.verbose > 1:
            print("Calculating MD-EI v2 acquisition (iterative MaxMin) ...", end=" ")

        # ──────────────────────────────────────────────────────────────
        # Step 0: Prepare candidate list and validate fingerprints
        # ──────────────────────────────────────────────────────────────
        xs_list = list(xs)

        # Accept arrays, generators, or accidentally-passed bound methods
        if callable(fingerprints):
            fingerprints = fingerprints()
        if not isinstance(fingerprints, np.ndarray):
            fingerprints = np.asarray(list(fingerprints))

        if len(fingerprints) != len(xs_list):
            raise ValueError(
                "MD-EI requires one fingerprint per candidate molecule: "
                f"got {len(fingerprints)} fingerprints for {len(xs_list)} candidates."
            )

        # ──────────────────────────────────────────────────────────────
        # Step 1: Compute Expected Improvement once (it doesn't depend on
        #         which molecules we pick within this batch)
        # ──────────────────────────────────────────────────────────────
        E_imp = metrics.ei(Y_mean, Y_var, current_max, self.xi)

        # Rank-based normalization: maps EI to a uniform [0, 1] distribution.
        # This prevents the long-tailed EI distribution from being crushed
        # by max-normalization (where 99% of molecules end up near 0).
        E_imp_normalized = metrics.rankdata(E_imp) / len(E_imp)

        # ──────────────────────────────────────────────────────────────
        # Step 2: Build "unexplored mask" — molecules NOT yet labeled
        # ──────────────────────────────────────────────────────────────
        unexplored_mask = np.ones(len(xs_list), dtype=bool)
        for i, x in enumerate(xs_list):
            if x in explored:
                unexplored_mask[i] = False

        # ──────────────────────────────────────────────────────────────
        # Step 3: KEY IMPROVEMENT — initialize running min-distance array
        #         using the TRAINING SET as the initial "selected set"
        # ──────────────────────────────────────────────────────────────
        # Every already-explored molecule counts as "already selected" for
        # diversity purposes. This means the very first pick of this batch
        # will already prefer regions far from the training set.
        train_indices = np.where(~unexplored_mask)[0]

        if len(train_indices) > 0:
            train_fps = fingerprints[train_indices]
            # running_min_dist[i] = min Tanimoto distance from candidate i
            # to ANY molecule in the current "selected set" (initially the
            # training set, later also includes batch picks)
            running_min_dist = metrics.batch_min_tanimoto_distance(
                fingerprints, train_fps
            )
        else:
            # First-ever iteration of MolPAL with no training data —
            # diversity signal is meaningless, fall back to max diversity
            running_min_dist = np.ones(len(xs_list), dtype=float)

        # ──────────────────────────────────────────────────────────────
        # Step 4: Track what we've selected in THIS batch
        # ──────────────────────────────────────────────────────────────
        selected_indices = []
        selected_molecules = []

        # ──────────────────────────────────────────────────────────────
        # Step 5: Iterative greedy MaxMin selection
        # ──────────────────────────────────────────────────────────────
        for _ in tqdm(range(batch_size), "MD-EI v2 Selection",
                    disable=self.verbose < 1):
            if not np.any(unexplored_mask):
                break  # No more unexplored molecules

            # Combine exploitation (EI) and exploration (diversity).
            # Both terms are now in [0, 1] and have comparable variance.
            U = (
                self.diversity_alpha * E_imp_normalized
                + (1 - self.diversity_alpha) * running_min_dist
            )

            # Mask out anything we can't pick
            U[~unexplored_mask] = -np.inf

            # Pick the argmax — the molecule that best balances
            # "high EI" and "far from everything already selected"
            best_idx = int(np.argmax(U))

            selected_indices.append(best_idx)
            selected_molecules.append(xs_list[best_idx])
            unexplored_mask[best_idx] = False

            # ──────────────────────────────────────────────────────────
            # Step 6: Update the running min-distance array.
            # Only one new molecule was added, so for each candidate we
            # just check whether the distance to this NEW molecule is
            # smaller than its current min-distance, and update if so.
            # This is O(n) per iteration, no recomputation needed.
            # ──────────────────────────────────────────────────────────
            new_distances = metrics.batch_min_tanimoto_distance(
                fingerprints, fingerprints[best_idx : best_idx + 1]
            )
            running_min_dist = np.minimum(running_min_dist, new_distances)

        if self.verbose > 1:
            print("Done!")
        if self.verbose > 2:
            total = default_timer() - begin
            mins, secs = divmod(int(total), 60)
            print(f"      MD-EI v2 acquisition took {mins}m {secs}s")
            print(f"      Selected {len(selected_molecules)} diverse molecules")

        return selected_molecules

    def scale_heaps(self, d_cid_heap: Dict[int, List], global_pred_max: float, it: int):
        """Scale each heap's size based on a decay factor

        The decay factor is calculated by an exponential decay based on the
        difference between a given heap's local maximum and the predicted
        global maximum then scaled by the current temperature. The temperature
        is also an exponential decay based on the current iteration starting at
        the initial temperature and approaching the final temperature.

        Parameters
        ----------
        d_cid_heap : Dict[int, List]
            a mapping from cluster_id to the heap containing the inputs to
            acquire from that cluster
        pred_global_max : float
            the predicted maximum value of the objective function
        it : int
            the current iteration of acquisition
        temp_i : float
            the initial temperature of the system
        temp_f : float
            the final temperature of the system

        Returns
        -------
        d_cid_heap
            the original mapping scaled down by the calculated decay factor
        """
        temp = Acquirer.temp(it, self.temp_i, self.temp_f)

        for cid, (heap, heap_size) in d_cid_heap.items():
            if len(heap) == 0:
                continue

            pred_local_max = max(heap, key=lambda yx: -1 if math.isinf(yx[0]) else yx[0])
            lam = Acquirer.decay(global_pred_max, pred_local_max, temp)

            new_heap_size = math.ceil(lam * heap_size)
            new_heap = heapq.nlargest(new_heap_size, heap)

            d_cid_heap[cid] = (new_heap, new_heap_size)

        return d_cid_heap

    @staticmethod
    def temp(it: int, temp_i, temp_f) -> float:
        """Calculate the temperature of the system"""
        return (temp_i - temp_f) * math.exp(-it) + temp_f

    @staticmethod
    def decay(global_max: float, local_max: float, temp: float) -> float:
        """Calculate the decay factor lambda of a given heap"""
        return math.exp(-(global_max - local_max) / temp)

